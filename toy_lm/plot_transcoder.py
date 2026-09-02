"""Plot a transcoder run from its metrics .log -- safe to call mid-run.

    python plot_transcoder.py                      # newest transcoder_*.log
    python plot_transcoder.py --log logs/transcoder_20260902_120543.log
    python plot_transcoder.py --ema-span 100

The same three-panel figure plot.py draws for the transformer, pointed at
transcoder_config.yaml so the clip and warmup guides come from the transcoder's
own hyperparameters.
"""

import argparse
from pathlib import Path

import plot

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "transcoder_config.yaml"


def latest_log():
    """The newest transcoder log. logs/latest.log is not it -- that symlink
    points at whichever run wrote last, transformer or transcoder.
    """
    logs = sorted(LOG_DIR.glob("transcoder_*.log"))
    if not logs:
        raise SystemExit(f"no transcoder_*.log in {LOG_DIR} -- has a run started?")
    return logs[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=None,
                   help="default: newest transcoder_*.log")
    p.add_argument("--out", type=Path, default=None, help="default: <log>.png")
    p.add_argument("--ema-span", type=int, default=50)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    # where render() reads max_grad_norm and warmup_steps for its guide lines
    plot.CONFIG_PATH = CONFIG_PATH
    plot.render(args.log or latest_log(), args.out, args.ema_span, args.dpi,
                ylabel="mse + l1")


if __name__ == "__main__":
    main()
