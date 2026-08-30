import random

class Dataset:
    def __init__(self, num_features, sparsity, num_samples, batch_size):
        self.num_features = num_features
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.batch_size = batch_size

    def build_sample(self):
        return [
            0.0 if random.random() < self.sparsity else random.random()
            for _ in range(self.num_features)
        ]

    def __iter__(self):
        batch = []

        for _ in range(self.num_samples):
            batch.append(self.build_sample())

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch