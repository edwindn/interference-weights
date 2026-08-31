"""Sample from a checkpoint and write each generation to generations/generation_i.txt.

Every knob comes from generation_config.yaml.
"""

import random
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer

from model import Transformer
from utils import pick_device

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
CONFIG_PATH = ROOT / "generation_config.yaml"
EOS = "<|end_of_text|>"  # the separator dataloader.py appends between documents


@torch.no_grad()
def generate(model, tokens, max_new_tokens, eos_id, temperature, top_k):
    """Returns the token ids and whether the run stopped on eos rather than the cap."""
    for _ in range(max_new_tokens):
        logits = model(tokens=tokens[:, -model.max_seq_len:])[:, -1, :]
        logits = logits.float() / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, nxt], dim=1)
        if nxt.item() == eos_id:
            return tokens, True
    return tokens, False


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = ROOT / cfg["checkpoint"]
    out_dir = ROOT / cfg["out_dir"]
    num = cfg["num_generations"]
    max_new_tokens = cfg["max_new_tokens"]

    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path}")

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = pick_device()

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id(EOS)
    if eos_id is None:
        raise SystemExit(f"tokenizer has no {EOS}, generations could never stop early")

    # weights_only would reject the Path objects the checkpoint keeps in "args"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["config"]
    model = Transformer(tokenizer.get_vocab_size(), model_cfg["embedding_dim"],
                        model_cfg["num_heads"], model_cfg["seq_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | {ckpt_path} | {num} generations -> {out_dir}")

    prompt_ids = tokenizer.encode(cfg["prompt"]).ids
    if not prompt_ids:
        raise SystemExit("prompt encodes to no tokens, give the model something to start from")

    stopped_on_eos = 0
    for i in range(1, num + 1):
        prompt = torch.tensor([prompt_ids], device=device)
        tokens, hit_eos = generate(model, prompt, max_new_tokens, eos_id,
                                   cfg["temperature"], cfg["top_k"])
        text = tokenizer.decode(tokens[0].tolist())

        path = out_dir / f"generation_{i}.txt"
        path.write_text(text)

        new = tokens.size(1) - len(prompt_ids)
        stopped_on_eos += hit_eos
        print(f"  {path.name}  {new} new tokens, {'eos' if hit_eos else 'cap'}")

    print(f"{stopped_on_eos}/{num} stopped on eos")


if __name__ == "__main__":
    main()
