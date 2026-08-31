"""Plot a training run from its metrics .log -- safe to call mid-run.

    python plot.py                  # logs/latest.log -> logs/latest.png
    python plot.py --ema-span 100
    python plot.py --log logs/train_20260830_214426.log

Raw per-step values are drawn faded; an EMA rides on top in full strength.
Loss and gradient norm sit side by side in one PNG -- unless the log carries no
gradient norms (an eval run has none), in which case loss gets the whole figure.

`render()` is the entry point eval.py reuses.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write a file, never open a window

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "logs" / "latest.log"
CONFIG_PATH = ROOT / "config.yaml"

# Validated categorical slots 1-3 (see the data-viz palette); one hue per entity.
TRAIN = "#2a78d6"   # blue
VAL = "#eb6834"     # orange
GNORM = "#1baf7a"   # aqua

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

RAW_ALPHA = 0.28    # the faded true-data trace
LINE_W = 2.0        # EMA on top


def read_log(path):
    """Tolerates a partially written trailing row while a run is in flight."""
    df = pd.read_csv(path, on_bad_lines="skip")
    return df.dropna(subset=["step", "train_loss"])


def ema_span(requested, n):
    """Shrink the span on short runs so the EMA still tracks the data."""
    return max(2, min(requested, n // 5)) if n < requested * 5 else requested


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def clip_threshold():
    """max_grad_norm from config.yaml, for the reference line. None if unreadable."""
    try:
        with open(CONFIG_PATH) as f:
            value = yaml.safe_load(f)["training"]["max_grad_norm"]
        return value if value and value > 0 else None
    except (OSError, KeyError, TypeError):
        return None


def style(ax, title, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")  # hairline, solid, recessive
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_facecolor(SURFACE)


def end_label(ax, x, y, text, color):
    """Visible value at the line end -- also the relief for low-contrast hues."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=TEXT, fontsize=8, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1.0))


def render(log, out=None, ema_span_hint=50, dpi=150,
           train_name="train", val_name="val"):
    """Draw `log` to a PNG and return its path. Names label the two loss series."""
    log = Path(log)
    if not log.exists():
        raise SystemExit(f"no log at {log} -- has a run started?")

    df = read_log(log)
    if df.empty:
        raise SystemExit(f"{log} has no rows yet")

    span = ema_span(ema_span_hint, len(df))
    out = Path(out) if out else log.with_suffix(".png")

    # An eval run logs no gradients, so that panel would be empty -- drop it.
    has_grad = "grad_norm" in df and df.grad_norm.notna().any()
    width = 11 if has_grad else 6.5
    fig, axes = plt.subplots(1, 2 if has_grad else 1, figsize=(width, 4.2),
                             facecolor=SURFACE, squeeze=False)
    ax_loss = axes[0, 0]

    # -- loss ---------------------------------------------------------------
    ax_loss.plot(df.step, df.train_loss, color=TRAIN, alpha=RAW_ALPHA, linewidth=1.0)
    smoothed = ema(df.train_loss, span)
    ax_loss.plot(df.step, smoothed, color=TRAIN, linewidth=LINE_W,
                 solid_joinstyle="round", solid_capstyle="round",
                 label=f"{train_name} (EMA {span})")

    val = df.dropna(subset=["val_loss"]) if "val_loss" in df else df.iloc[0:0]
    if not val.empty:
        ax_loss.plot(val.step, val.val_loss, color=VAL, linewidth=LINE_W,
                     marker="o", markersize=4.5, label=val_name)

    style(ax_loss, "Loss", "cross-entropy")
    end_label(ax_loss, df.step.iloc[-1], smoothed.iloc[-1], f"{smoothed.iloc[-1]:.3f}", TRAIN)
    if not val.empty:
        end_label(ax_loss, val.step.iloc[-1], val.val_loss.iloc[-1],
                  f"{val.val_loss.iloc[-1]:.3f}", VAL)
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

    # -- gradient norm ------------------------------------------------------
    if has_grad:
        ax_grad = axes[0, 1]
        ax_grad.plot(df.step, df.grad_norm, color=GNORM, alpha=RAW_ALPHA, linewidth=1.0)
        smoothed_g = ema(df.grad_norm, span)
        ax_grad.plot(df.step, smoothed_g, color=GNORM, linewidth=LINE_W,
                     solid_joinstyle="round", solid_capstyle="round",
                     label=f"grad norm (EMA {span})")

        clip = clip_threshold()
        if clip is not None:
            ax_grad.axhline(clip, color=MUTED, linewidth=1.0, alpha=0.6,
                            label=f"clip at {clip:g}")

        style(ax_grad, "Gradient norm (pre-clip)", "L2 norm")
        end_label(ax_grad, df.step.iloc[-1], smoothed_g.iloc[-1],
                  f"{smoothed_g.iloc[-1]:.2f}", GNORM)
        ax_grad.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

    last = df.step.iloc[-1]
    fig.suptitle(f"{log.name}  ·  {len(df)} steps logged (through step {last:.0f})",
                 color=MUTED, fontsize=9, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    print(f"{out}  ({len(df)} rows, EMA span {span})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--out", type=Path, default=None, help="default: <log>.png")
    p.add_argument("--ema-span", type=int, default=50)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    render(args.log, args.out, args.ema_span, args.dpi)


if __name__ == "__main__":
    main()
