"""Single-pass streaming batches from PleIAs/common_corpus.

`load_dataset(streaming=True)` is not incremental on this corpus: every shard is
a single ~430MB row group, so the reader decodes the whole thing and sits at a
~3.8GB floor. Instead each shard is fetched to disk once and read memory-mapped
in record batches, which holds the reader near 0.3GB.

Two background threads keep the training loop fed: one downloads the next shard
while the current one is being consumed, the other runs the tokeniser ahead of
the GPU. No epochs -- every token is served exactly once.
"""

import queue
import threading

import torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "PleIAs/common_corpus"
CODE_COLLECTION = "Github Open Source"
END_OF_TEXT = "<|end_of_text|>"

COLUMNS = ["language", "collection", "text"]
ROW_BATCH = 256      # rows decoded per arrow batch
ENCODE_CHUNK = 16    # docs per tokeniser call


class CorpusExhausted(Exception):
    """Raised by `batch()` once the shards have been consumed."""


def shard_files(num_shards=10, subset="subset_100_1.parquet"):
    """One shard from each top-level folder, for spread across the corpus."""
    if not 1 <= num_shards <= 10:
        raise ValueError(f"num_shards must be in 1..10, got {num_shards}")
    return [f"common_corpus_{i}/{subset}" for i in range(1, num_shards + 1)]


def english_only(row):
    return row["language"] == "English"


def code_only(row):
    return row["collection"] == CODE_COLLECTION


FILTERS = {"english": english_only, "code": code_only}


class StreamingCorpus:
    """Serves (input, target) windows with the same interface as `TextData`."""

    def __init__(self, tokenizer, seq_len, batch_size, device, mode="english",
                 num_shards=10, val_batches=20, prefetch=32, files=None):
        if mode not in FILTERS:
            raise ValueError(f"mode must be one of {sorted(FILTERS)}, got {mode!r}")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.keep = FILTERS[mode]
        self.files = files or shard_files(num_shards)
        self.eod = tokenizer.token_to_id(END_OF_TEXT)

        self.window = seq_len + 1
        self.tokens_seen = 0
        self.docs_kept = 0

        self._stop = threading.Event()
        self._error = None

        # depth 1: fetch the next shard while the current one is being read
        self._paths = queue.Queue(maxsize=1)
        self._batches = queue.Queue(maxsize=prefetch)
        self._threads = [
            threading.Thread(target=self._download, daemon=True),
            threading.Thread(target=self._produce, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

        # Held out before any training batch is drawn, so the two never overlap.
        self._val = [self._next_batch() for _ in range(val_batches)]
        self._val_pos = 0

    def _download(self):
        """Fetch shards to disk, one ahead of the reader."""
        try:
            for name in self.files:
                if self._stop.is_set():
                    break
                path = hf_hub_download(REPO, name, repo_type="dataset")
                while not self._stop.is_set():
                    try:
                        self._paths.put(path, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # surfaced to the consumer on the next batch()
            self._error = exc
        self._paths.put(None)

    def _rows(self):
        """Yield the text of matching documents, one shard at a time."""
        while not self._stop.is_set():
            path = self._paths.get()
            if path is None:
                return
            reader = pq.ParquetFile(path, memory_map=True)
            for arrow_batch in reader.iter_batches(batch_size=ROW_BATCH, columns=COLUMNS):
                if self._stop.is_set():
                    return
                for row in arrow_batch.to_pylist():
                    if row["text"] and self.keep(row):
                        self.docs_kept += 1
                        yield row["text"]

    def _produce(self):
        """Tokenise documents into a rolling buffer and emit fixed-size batches."""
        buffer = []
        windows = []
        try:
            for texts in self._chunks(self._rows(), ENCODE_CHUNK):
                for encoding in self.tokenizer.encode_batch(texts):
                    buffer.extend(encoding.ids)
                    if self.eod is not None:
                        buffer.append(self.eod)

                while len(buffer) >= self.window:
                    windows.append(buffer[:self.window])
                    del buffer[:self.window]
                    if len(windows) == self.batch_size:
                        if not self._put(windows):
                            return
                        windows = []
        except Exception as exc:  # surfaced to the consumer on the next batch()
            self._error = exc
        self._batches.put(None)

    @staticmethod
    def _chunks(iterable, size):
        chunk = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _put(self, windows):
        """Queue one batch as CPU tensors. Returns False once the consumer stops."""
        block = torch.tensor(windows, dtype=torch.long)
        while not self._stop.is_set():
            try:
                self._batches.put((block[:, :-1], block[:, 1:]), timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _next_batch(self):
        item = self._batches.get()
        if item is None:
            if self._error is not None:
                raise self._error
            raise CorpusExhausted(f"consumed {len(self.files)} shard(s)")
        inputs, targets = item
        self.tokens_seen += inputs.numel()
        return inputs, targets

    def batch(self, split="train"):
        if split == "val":
            inputs, targets = self._val[self._val_pos % len(self._val)]
            self._val_pos += 1
        else:
            inputs, targets = self._next_batch()
        return inputs.to(self.device), targets.to(self.device)

    def close(self):
        self._stop.set()
