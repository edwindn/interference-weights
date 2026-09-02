"""Plot a training run from its metrics .log -- safe to call mid-run.

    python plot.py                  # logs/latest.log -> logs/latest.png
    python plot.py --ema-span 100
    python plot.py --log logs/train_20260830_214426.log

Raw per-step values are drawn faded; an EMA rides on top in full strength.
Loss, gradient norm and learning rate sit side by side in one PNG. A panel is
dropped when the log has no such column, so an eval run -- which records neither
gradients nor lr -- leaves loss the whole figure.

`render()` is the entry point eval.py reuses.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_LOG = ROOT / "logs" / "latest.log"
CONFIG_PATH = ROOT / "config.yaml"

TRAIN = "#2a78d6"
VAL = "#eb6834"
GNORM = "#1baf7a"
LR = "#8f5bd6"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

RAW_ALPHA = 0.28
LINE_W = 2.0


def read_log(path):
    """Tolerates a partially written trailing row while a run is in flight."""
    df = pd.read_csv(path, on_bad_lines="skip")
    return df.dropna(subset=["step", "train_loss"])


def ema_span(requested, n):
    """Shrink the span on short runs so the EMA still tracks the data."""
    return max(2, min(requested, n // 5)) if n < requested * 5 else requested


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def config_value(key):
    """A positive `training` value from config.yaml, or None if unreadable."""
    try:
        with open(CONFIG_PATH) as f:
            value = yaml.safe_load(f)["training"][key]
        return value if value and value > 0 else None
    except (OSError, KeyError, TypeError):
        return None


def style(ax, title, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("step", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_facecolor(SURFACE)


def end_label(ax, x, y, text, color):
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                color=TEXT, fontsize=8, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1.0))


def render(log, out=None, ema_span_hint=50, dpi=150,
           train_name="train", val_name="val", ylabel="cross-entropy"):
    """Draw `log` to a PNG and return its path. Names label the two loss series."""
    log = Path(log)
    if not log.exists():
        raise SystemExit(f"no log at {log} -- has a run started?")

    df = read_log(log)
    if df.empty:
        raise SystemExit(f"{log} has no rows yet")

    span = ema_span(ema_span_hint, len(df))
    out = Path(out) if out else log.with_suffix(".png")

    has_grad = "grad_norm" in df and df.grad_norm.notna().any()
    has_lr = "lr" in df and df.lr.notna().any()
    panels = 1 + has_grad + has_lr
    fig, axes = plt.subplots(1, panels, figsize=(6.5 if panels == 1 else 5.5 * panels, 4.2),
                             facecolor=SURFACE, squeeze=False)
    ax_loss = axes[0, 0]
    nxt = 1

    ax_loss.plot(df.step, df.train_loss, color=TRAIN, alpha=RAW_ALPHA, linewidth=1.0)
    smoothed = ema(df.train_loss, span)
    ax_loss.plot(df.step, smoothed, color=TRAIN, linewidth=LINE_W,
                 solid_joinstyle="round", solid_capstyle="round",
                 label=f"{train_name} (EMA {span})")

    val = df.dropna(subset=["val_loss"]) if "val_loss" in df else df.iloc[0:0]
    if not val.empty:
        ax_loss.plot(val.step, val.val_loss, color=VAL, linewidth=LINE_W,
                     marker="o", markersize=4.5, label=val_name)

    style(ax_loss, "Loss", ylabel)
    end_label(ax_loss, df.step.iloc[-1], smoothed.iloc[-1], f"{smoothed.iloc[-1]:.3f}", TRAIN)
    if not val.empty:
        end_label(ax_loss, val.step.iloc[-1], val.val_loss.iloc[-1],
                  f"{val.val_loss.iloc[-1]:.3f}", VAL)
    ax_loss.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

    if has_grad:
        ax_grad = axes[0, nxt]
        nxt += 1
        ax_grad.plot(df.step, df.grad_norm, color=GNORM, alpha=RAW_ALPHA, linewidth=1.0)
        smoothed_g = ema(df.grad_norm, span)
        ax_grad.plot(df.step, smoothed_g, color=GNORM, linewidth=LINE_W,
                     solid_joinstyle="round", solid_capstyle="round",
                     label=f"grad norm (EMA {span})")

        clip = config_value("max_grad_norm")
        if clip is not None:
            ax_grad.axhline(clip, color=MUTED, linewidth=1.0, alpha=0.6,
                            label=f"clip at {clip:g}")

        style(ax_grad, "Gradient norm (pre-clip)", "L2 norm")
        end_label(ax_grad, df.step.iloc[-1], smoothed_g.iloc[-1],
                  f"{smoothed_g.iloc[-1]:.2f}", GNORM)
        ax_grad.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

    if has_lr:
        ax_lr = axes[0, nxt]
        ax_lr.plot(df.step, df.lr, color=LR, linewidth=LINE_W,
                   solid_joinstyle="round", solid_capstyle="round", label="learning rate")

        warmup = config_value("warmup_steps")
        if warmup is not None and df.step.iloc[-1] >= warmup:
            ax_lr.axvline(warmup, color=MUTED, linewidth=1.0, alpha=0.6,
                          label=f"warmup ends at {warmup:g}")

        ax_lr.set_ylim(bottom=0)
        style(ax_lr, "Learning rate", "lr")
        ax_lr.ticklabel_format(style="sci", scilimits=(0, 0), axis="y")
        ax_lr.yaxis.get_offset_text().set_color(MUTED)
        ax_lr.yaxis.get_offset_text().set_fontsize(8)
        end_label(ax_lr, df.step.iloc[-1], df.lr.iloc[-1], f"{df.lr.iloc[-1]:.2e}", LR)
        ax_lr.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")

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
