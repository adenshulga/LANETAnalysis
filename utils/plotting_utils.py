import numpy as np

from typing import Tuple

import matplotlib.pyplot as plt


def plot_label_distribution(
    gt: np.ndarray, *args: np.ndarray, figsize: Tuple[int, int] = (18, 10)
) -> None:
    """this function supports plotting of distribution of multiple arrays"""
    # Calculate the number of labels
    num_labels = gt.shape[1]
    print(num_labels)
    # Calculate label occurrences for each dataset
    occurrences = [gt.sum(axis=0) / gt.shape[0]]

    for arg in args:
        occurrences.append(arg.sum(axis=0) / arg.shape[0])

    # Determine the number of groups and bar width
    n_groups = num_labels
    n_labels = len(occurrences)
    bar_width = 1 / (n_labels + 1)
    # Create figure
    plt.figure(figsize=figsize)
    # Create an index for each group
    index = np.arange(n_groups)
    # Plot each dataset
    for i, occ in enumerate(occurrences):
        plt.bar(index + i * bar_width, occ, bar_width, label=f"Dataset {i}", alpha=0.5)

    plt.xlabel("Label Index")
    plt.ylabel("Frequency")
    plt.title("Distribution of Label Frequencies")
    plt.xticks(index + bar_width / 2 * (n_labels - 1), index)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_probas_distribution(probas: np.ndarray, index_of_label: int) -> None:
    plt.hist(probas[:, index_of_label])
    plt.title(f"Distribution of predicted probas for label {index_of_label}")

    plt.show()
