from typing import List

import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        ensure_all_classes_in_batch: bool = False,
    ):
        """
        Args:
            labels (List[int]): A list of class labels for each sample in the dataset.
            batch_size (int): The size of each batch to return.
            ensure_all_classes_in_batch (bool): If True, ensures that
                            each batch contains at least one sample from each class.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.ensure_all_classes_in_batch = ensure_all_classes_in_batch

        self.num_classes = len(np.unique(self.labels))
        self.class_indices = self._get_class_indices()

        if self.ensure_all_classes_in_batch:
            assert self.batch_size >= self.num_classes, (
                "Batch size must be greater than or equal to the number of classes "
                "when ensure_all_classes_in_batch is True."
            )

    def _get_class_indices(self):
        """Group indices by class labels."""
        class_indices = {}
        for cls_idx in np.unique(self.labels):
            indices = np.where(self.labels == cls_idx)[0]
            class_indices[cls_idx] = indices.tolist()
        return class_indices

    def __iter__(self):
        if self.ensure_all_classes_in_batch:
            return self._iter_balanced()
        else:
            return self._iter_weighted_random()

    def _iter_balanced(self):
        """Yield batches ensuring all classes are present in each batch."""
        # Shuffle indices within each class
        for indices in self.class_indices.values():
            np.random.shuffle(indices)

        # Calculate how many samples per class per batch
        samples_per_class = self.batch_size // self.num_classes
        remainder = self.batch_size % self.num_classes

        # Prepare pointers to class indices
        class_pointers = {cls_idx: 0 for cls_idx in self.class_indices}

        batch = []
        while True:
            for cls_idx in self.class_indices:
                start = class_pointers[cls_idx]
                end = start + samples_per_class
                indices = self.class_indices[cls_idx][start:end]
                batch.extend(indices)
                class_pointers[cls_idx] += samples_per_class

            # Distribute the remainder samples randomly among classes
            if remainder > 0:
                extra_classes = np.random.choice(
                    list(self.class_indices.keys()), size=remainder, replace=False
                )
                for cls_idx in extra_classes:
                    start = class_pointers[cls_idx]
                    end = start + 1
                    indices = self.class_indices[cls_idx][start:end]
                    batch.extend(indices)
                    class_pointers[cls_idx] += 1

            if len(batch) == 0:
                break

            yield from batch
            batch = []

            # Check if any class has exhausted its samples
            exhausted = any(
                class_pointers[cls_idx] >= len(self.class_indices[cls_idx])
                for cls_idx in self.class_indices
            )
            if exhausted:
                break

    def _iter_weighted_random(self):
        """Yield batches using weighted random sampling."""
        # Compute class weights
        class_counts = np.array(
            [len(self.class_indices[cls_idx]) for cls_idx in self.class_indices]
        )
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[label] for label in self.labels])

        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights)

        # Generate indices
        indices = np.random.choice(
            len(self.labels),
            size=len(self.labels),
            replace=True,
            p=sample_weights,
        )

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            yield from indices[i : i + self.batch_size]

    def __len__(self):
        return len(self.labels)
