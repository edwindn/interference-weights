"""How much of shard 1 survives a tokenizer truncated to its first N ids?

Two different questions, both answered here:

  as-is       encode with the full 65536-token tokenizer, then ask whether every
              id landed below the cut. This is what "the model only ever needs
              the low ids" would mean, and it is the pessimistic reading.
  restricted  rebuild the tokenizer keeping only ids < N (and only the merges
              whose parents and result all survive), re-encode, and ask whether
              the text comes back byte-identical. Lossless as long as the cut
              keeps the 256-char ByteLevel alphabet, at the cost of longer
              sequences -- so the number that matters here is the inflation.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
CODE_COLLECTION = "Github Open Source"
COLUMNS = ["language", "collection", "text"]
THRESHOLDS = [1024, 2048, 4096, 8192, 16384, 32768]


def restricted_tokenizer(spec, limit, force_bytes=False):
    """Same tokenizer with every id >= limit removed.

    A plain prefix cut drops 13 of the 256 ByteLevel alphabet chars, which leaves
    those byte values with no representation at all. force_bytes buys them back
    by evicting the least frequent surviving merges, keeping the budget at limit.
    """
    spec = json.loads(json.dumps(spec))  # deep copy, spec is reused per threshold
    spec["model"]["vocab"] = {s: i for s, i in spec["model"]["vocab"].items() if i < limit}
    if force_bytes:
        full = json.loads(TOKENIZER_PATH.read_text())["model"]["vocab"]
        needed = [c for c in byte_alphabet().values() if c not in spec["model"]["vocab"]]
        # highest id == rarest, so drop those first; never evict a single char
        evictable = sorted((i, s) for s, i in spec["model"]["vocab"].items() if len(s) > 1)
        for (_, victim), char in zip(reversed(evictable), needed):
            spec["model"]["vocab"].pop(victim)
        spec["model"]["vocab"].update({c: full[c] for c in needed})
        # ids must be a dense 0..n-1 range for the merge table to stay valid
        spec["model"]["vocab"] = {
            s: n for n, (s, _) in enumerate(sorted(spec["model"]["vocab"].items(),
                                                   key=lambda kv: kv[1]))}
    kept = spec["model"]["vocab"]
    # A merge is only usable if both parents and the token it produces survived.
    spec["model"]["merges"] = [
        m for m in spec["model"]["merges"]
        if m[0] in kept and m[1] in kept and (m[0] + m[1]) in kept
    ]
    spec["added_tokens"] = [a for a in spec.get("added_tokens", []) if a["id"] < limit]
    return Tokenizer.from_str(json.dumps(spec))


def byte_alphabet():
    """The 256 single-char tokens a ByteLevel pre-tokenizer can emit (GPT-2 map)."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    table, spare = printable[:], 0
    out = {}
    for b in range(256):
        if b in table:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + spare)
            spare += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--limit", type=int, default=4096, help="vocab cut to study in depth")
    p.add_argument("--max-docs", type=int, default=0, help="0 = whole shard")
    p.add_argument("--row-batch", type=int, default=256)
    p.add_argument("--sample", type=float, default=1.0, help="fraction of rows to keep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-bytes", action="store_true",
                   help="keep the full 256-char byte alphabet inside the budget")
    args = p.parse_args()

    spec = json.loads(TOKENIZER_PATH.read_text())
    full = Tokenizer.from_file(str(TOKENIZER_PATH))
    cut = restricted_tokenizer(spec, args.limit, args.force_bytes)

    cut_vocab = cut.get_vocab()
    missing = [c for c in byte_alphabet().values() if c not in cut_vocab]
    print(f"tokenizer: {full.get_vocab_size()} ids, cut to {args.limit}")
    print(f"  merges kept: {len(json.loads(cut.to_str())['model']['merges'])} "
          f"of {len(spec['model']['merges'])}")
    print(f"  ByteLevel alphabet chars missing below the cut: {len(missing)}"
          + (f" -> {missing[:8]} (encoding WILL be lossy)" if missing else
             " -> every byte representable, encoding is lossless"))
    print()

    rng = random.Random(args.seed)
    reader = pq.ParquetFile(args.shard, memory_map=True)

    # counters, keyed by group
    n = Counter()
    asis_ok = {t: Counter() for t in THRESHOLDS}
    tok_below = {t: Counter() for t in THRESHOLDS}
    tok_total = Counter()
    cut_lossless = Counter()
    cut_unk = Counter()
    len_full = Counter()
    len_cut = Counter()
    docs = 0

    def groups_of(row):
        g = ["all"]
        if row["language"] == "English":
            g.append("english")
        if row["collection"] == CODE_COLLECTION:
            g.append("code")
        if len(g) > 1:
            g.append("train_filter")  # what main.py actually streams
        else:
            g.append("other")
        return g

    for arrow_batch in reader.iter_batches(batch_size=args.row_batch, columns=COLUMNS):
        rows = [r for r in arrow_batch.to_pylist() if r["text"]]
        if args.sample < 1.0:
            rows = [r for r in rows if rng.random() < args.sample]
        if not rows:
            continue
        if args.max_docs and docs + len(rows) > args.max_docs:
            rows = rows[: args.max_docs - docs]
        texts = [r["text"] for r in rows]

        enc_full = full.encode_batch(texts)
        enc_cut = cut.encode_batch(texts)
        dec_cut = cut.decode_batch([e.ids for e in enc_cut])

        for row, ef, ec, dc in zip(rows, enc_full, enc_cut, dec_cut):
            gs = groups_of(row)
            ids = ef.ids
            hi = max(ids) if ids else 0
            for g in gs:
                n[g] += 1
                tok_total[g] += len(ids)
                len_full[g] += len(ids)
                len_cut[g] += len(ec.ids)
                for t in THRESHOLDS:
                    if hi < t:
                        asis_ok[t][g] += 1
                    tok_below[t][g] += sum(1 for i in ids if i < t)
                if 0 in ec.ids:
                    cut_unk[g] += 1
                elif dc == row["text"]:
                    cut_lossless[g] += 1
        docs += len(rows)
        if docs % 5120 < args.row_batch:
            print(f"  ...{docs} docs, {tok_total['all']/1e6:.1f}M tokens", flush=True)
        if args.max_docs and docs >= args.max_docs:
            break

    order = ["all", "train_filter", "english", "code", "other"]
    print(f"\ndocs scanned: {docs}   tokens (full vocab): {tok_total['all']/1e6:.1f}M\n")

    print(f"{'group':<14}{'docs':>9}", end="")
    for t in THRESHOLDS:
        print(f"{t:>9}", end="")
    print()
    print("as-is: % of docs whose full-vocab encoding stays entirely below the cut")
    for g in order:
        if not n[g]:
            continue
        print(f"{g:<14}{n[g]:>9}", end="")
        for t in THRESHOLDS:
            print(f"{100*asis_ok[t][g]/n[g]:>8.2f}%", end="")
        print()
    print("\nas-is: % of all tokens that are below the cut")
    for g in order:
        if not n[g]:
            continue
        print(f"{g:<14}{n[g]:>9}", end="")
        for t in THRESHOLDS:
            print(f"{100*tok_below[t][g]/max(tok_total[g],1):>8.2f}%", end="")
        print()

    print(f"\nrestricted to first {args.limit} ids (re-encoded):")
    print(f"{'group':<14}{'docs':>9}{'lossless':>11}{'hit [UNK]':>11}"
          f"{'tok/doc full':>14}{'tok/doc cut':>13}{'inflation':>11}")
    for g in order:
        if not n[g]:
            continue
        print(f"{g:<14}{n[g]:>9}{100*cut_lossless[g]/n[g]:>10.2f}%"
              f"{100*cut_unk[g]/n[g]:>10.2f}%"
              f"{len_full[g]/n[g]:>14.0f}{len_cut[g]/n[g]:>13.0f}"
              f"{len_cut[g]/max(len_full[g],1):>10.2f}x")


if __name__ == "__main__":
    main()
