import random

import torch
import torch.nn.functional as F
import yaml

from toy import Toy
from data import Dataset
from utils import loss_func


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main(seed):
    config = load_config()

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    random.seed(seed)

    num_features = data_cfg["num_features"]
    importance_base = data_cfg["importance_base"]

    dataset = Dataset(
        num_features=num_features,
        sparsity=data_cfg["sparsity"],
        num_samples=train_cfg["num_samples"],
        batch_size=train_cfg["batch_size"],
    )

    model = Toy(
        in_dim=data_cfg["num_features"],
        hidden_dim=data_cfg["projected_dim"],
        init_method=model_cfg["init_method"],
        activation=model_cfg["activation"],
        activation_after=model_cfg["activation_after"],
        seed=seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    epochs = train_cfg["epochs"]
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataset:
            x = torch.tensor(batch, dtype=torch.float32)

            optimizer.zero_grad()
            xp, _ = model(x)
            loss = loss_func(xp, x, num_features, importance_base)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"epoch {epoch + 1}/{epochs}  loss {total_loss / num_batches:.6f}")

    CHECKPOINT_PATH = f"weights/model_seed_{seed}.pt"
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"saved model weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    NUM_SEEDS = 100
    for seed in range(1, NUM_SEEDS + 1):
        main(seed)
