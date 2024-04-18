import torch
from datasets import Dataset, DatasetDict, load_dataset, IterableDatasetDict

c4_dataset = load_dataset("allenai/c4", "en", streaming=True)

def save_subset_as_dataset(dataset: IterableDatasetDict, num_samples: int, subset_name: str) -> None:
    """
    Save a small subset of a Hugging Face iterable dataset as a separate Hugging Face dataset.

    Args:
        dataset (IterableDatasetDict): The Hugging Face iterable dataset to subset.
        num_samples (int): The number of samples to include in the subset.
        subset_name (str): The name of the subset dataset.
    """
    # Get the first element of the IterableDatasetDict (assuming there is only one dataset)
    dataset_name = list(dataset.keys())[0]
    iterable_dataset = dataset[dataset_name]

    # Create a subset by iterating over the dataset and retrieving the desired number of samples
    subset_data = []
    iterator = iter(iterable_dataset)
    for _ in range(num_samples):
        try:
            sample = next(iterator)
            subset_data.append(sample)
        except StopIteration:
            break

    # Create a Dataset from the subset data
    subset_dataset = Dataset.from_list(subset_data)

    # Save the subset dataset
    subset_dataset.save_to_disk(subset_name)

    print(f"Subset '{subset_name}' with {len(subset_data)} samples saved as a Hugging Face dataset.")

save_subset_as_dataset(c4_dataset, 111000, "c4_mini")
