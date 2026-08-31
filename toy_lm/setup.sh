#!/usr/bin/env bash
set -euo pipefail

REPO="PleIAs/Pleias-1.2b-Preview"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/tokenizer/Pleias-1.2b-Preview"
FILES=(tokenizer.json tokenizer_config.json special_tokens_map.json)

mkdir -p "$DEST"
for name in "${FILES[@]}"; do
    if [[ -s "$DEST/$name" && -z "${FORCE:-}" ]]; then
        echo "have     $name"
        continue
    fi
    echo "fetching $name"
    curl -fL --progress-bar -o "$DEST/$name.part" \
        "https://huggingface.co/$REPO/resolve/main/$name"
    mv "$DEST/$name.part" "$DEST/$name"
done

python3 - "$DEST/tokenizer.json" <<'PY'
import json, sys

path = sys.argv[1]
with open(path) as f:
    tok = json.load(f)
size = len(tok["model"]["vocab"])
print(f"\ntokenizer ok: {size} tokens -> {path}")
PY
