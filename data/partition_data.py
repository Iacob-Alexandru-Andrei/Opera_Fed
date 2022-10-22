from data import (
    apply_sampling,
    filtering_activities_and_label_encoding,
    prepare_one_dataloader,
)
from data.load_data import import_multiple_modalities, split_multimodal_data
from typing import Any
from pathlib import Path
import torch


def create_person_partitions(
    data_type,
    data_directory,
    fed_dir,
    person_names,
    namings,
    sampling,
    y_sampling,
    activities=[],
    views="associated",
    axis=3,
):
    train_paritions = []
    test_partitions = []
    samplings = []
    for idx, person in enumerate(person_names):
        multimodal_data = import_multiple_modalities(
            data_type=data_type,
            data_directory=data_directory,
            namings=namings,
            partition_str=person,
        )
        X_train, X_test, y_train, y_test = split_multimodal_data(
            multimodal_data, views=views, axis=axis
        )
        X_train, X_test, y_train, y_test, lb = filtering_activities_and_label_encoding(
            X_train, X_test, y_train, y_test, activities
        )
        X_train, X_test, y_train, y_test = apply_sampling(
            X_train, X_test, y_train, y_test, sampling, lb, y_sampling
        )
        train_parition = (X_train, y_train)
        test_partition = (X_test, y_test)

        save_partition(
            partition=train_parition, idx=idx, fed_dir=fed_dir, partition_type="train"
        )
        save_partition(
            partition=test_partition, idx=idx, fed_dir=fed_dir, partition_type="test"
        )
        save_partition(
            partition=sampling, idx=idx, fed_dir=fed_dir, partition_type="sampling"
        )


def get_partition_data(
    path_to_data: Path,
    cid: str,
    partition_type: str,
    batch_size,
    num_workers,
    ext="pt",
):
    client_dir = path_to_data / f"{cid}"
    X, y = torch.load(client_dir / f"{partition_type}.{ext}")
    sampling = torch.load(client_dir / f"sampling.{ext}")

    return prepare_one_dataloader(
        X_train=X,
        y_train=y,
        sampling=sampling,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def save_partitions(list_partitions: Any, fed_dir: Path, partition_type: str = "train"):
    """Saves partitions to individual files.

    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / f"{partition_type}.pt")


def save_partition(
    partition: Any, idx, fed_dir: Path, partition_type: str = "fed_test"
):
    """Saves partitions to individual files.

    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """

    path_dir = fed_dir / f"{idx}"
    path_dir.mkdir(exist_ok=True, parents=True)
    torch.save(partition, path_dir / f"{partition_type}.pt")
