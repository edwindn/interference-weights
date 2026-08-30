import argparse

import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CHECKPOINT_PATH = "weights/model.pt"
OUTPUT_PATH = "weights.png"

# diverging: blue <-> red poles with a neutral gray midpoint, so sign reads as
# direction and zero recedes into the surface
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red",
    [
        "#0d366b",
        "#2a78d6",
        "#cde2fb",
        "#f0efec",
        "#fbd6d5",
        "#e34948",
        "#7a1f1f",
    ],
)

SURFACE = "#ffffff"
INK = "#33322e"
MUTED = "#73726c"


def load_params(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "weight" not in state:
        raise ValueError(f"{path} does not look like a Toy state_dict")

    # stored as (num_features, hidden_dim); W is the encoder (hidden_dim, num_features)
    W = state["weight"].detach().float().T
    b = state["bias"].detach().float()
    return W, b


def tick_positions(n):
    stride = max(1, n // 20)
    return list(range(0, n, stride))


def plot(W, b, out_path):
    gram = (W.T @ W).numpy()
    bias = b.numpy()[:, None]
    n = gram.shape[0]

    # one symmetric scale shared by both panels, so a colour means the same
    # value in W.T @ W and in b
    lim = max(float(abs(gram).max()), float(abs(bias).max())) or 1.0

    fig = plt.figure(figsize=(8.0, 6.0), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.06, 0.045], wspace=0.28)
    ax_gram = fig.add_subplot(grid[0, 0])
    ax_bias = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])

    im_gram = ax_gram.imshow(
        gram, cmap=DIVERGING, vmin=-lim, vmax=lim, interpolation="nearest"
    )
    ax_gram.set_title("$W^{\\top}W$", color=INK, fontsize=12, pad=10)
    ax_gram.set_xlabel("feature", color=MUTED, fontsize=9)
    ax_gram.set_ylabel("feature", color=MUTED, fontsize=9)

    ticks = tick_positions(n)
    labels = [str(i + 1) for i in ticks]
    ax_gram.set_xticks(ticks, labels)
    ax_gram.set_yticks(ticks, labels)

    ax_bias.imshow(
        bias,
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        interpolation="nearest",
        aspect="auto",
    )
    ax_bias.set_title("$b$", color=INK, fontsize=12, pad=10)
    ax_bias.set_xticks([])
    ax_bias.set_yticks(ticks, labels)

    for ax in (ax_gram, ax_bias):
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    bar = fig.colorbar(im_gram, cax=cax)
    bar.outline.set_visible(False)
    # only the poles and the neutral midpoint carry a number
    bar.set_ticks([-lim, 0.0, lim], labels=[f"{-lim:.2g}", "0", f"{lim:.2g}"])
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return fig


def main():
    parser = argparse.ArgumentParser(description="visualize a Toy checkpoint")
    parser.add_argument("checkpoint", nargs="?", default=CHECKPOINT_PATH)
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    W, b = load_params(args.checkpoint)
    plot(W, b, args.out)
    print(f"wrote {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
