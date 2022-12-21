from curses import nonl
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History
from torch.nn import Module
from torch.utils.data import DataLoader
from train import test_client


from models.cnn import ResNet, Block, toy_model_generator
from models.hybridvit import *
from models.transformer import VisionTransformer
from simmim.vision_transformer import ViT
from models.cnn import ResNet
from collections import OrderedDict
import numpy as np
from data import prepare_one_dataloader
import matplotlib.pyplot as plt

from typing import List
import flwr as fl
from flwr.common import Metrics
from collections import defaultdict
import numbers
import torch.nn.functional as F
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters


def save_model(params, save_dir, save_name, ext="pt"):
    aggregate_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / f"{save_name}.{ext}",
        aggregate_ndarrays,
    )


def load_model(save_dir, save_name, ext="pt"):
    save_dir = Path(save_dir)
    weights = [
        arr
        for val in np.load(save_dir / f"{save_name}.{ext}", allow_pickle=True).values()
        for arr in val
    ]
    parameters = ndarrays_to_parameters(weights)
    return parameters


def wrapper_func(func: Callable) -> Callable[[], Callable]:
    """Decorator to wrap any function

    Args:
        func (Callable): Function to wrap

    Returns:
         Callable[[], Callable]: Boxed function
    """
    return lambda: func


@wrapper_func
def configure_data_noop(config: dict):
    return None


@wrapper_func
def configure_data_window(config: dict):
    def configure_data(data: tuple):
        nonlocal config
        X, y = data
        data_len = len(y)


def get_transform_noop():
    def transform_noop(tensor):
        return tensor

    return transform_noop


def get_transform_x_pad_spectogram_height(input_dim, axis):
    def transform_x(tensor):
        pad_size: int = max(tensor.shape[axis - 1], input_dim) - tensor.shape[axis - 1]
        return F.pad(tensor, [0, pad_size])

    return transform_x


def get_transform_x_pad_concat(input_dim, axis):
    def transform_x(tensor):

        pad_size: int = max(tensor.shape[axis - 1], input_dim) - tensor.shape[axis - 1]

        if tensor.shape[axis - 1] == 672:
            return F.pad(tensor, [0, pad_size])
        else:
            return F.pad(tensor, [pad_size, 0])

    return transform_x


def get_network(network, input_dim):
    if network == "vit":
        img_size = [224, input_dim]
        patch_size = [224, 224]  # [32,32]  [16,16]
        in_channels = 1
        n_classes = 6
        embed_dim = 512  # 384 # 512 #2048
        depth = 3  # 6  4
        n_heads = 4
        qkv_bias = False
        attn_p = 0.1
        p = 0.1
        mlp_ratio = 1.0

        model = VisionTransformer(
            img_size,
            patch_size,
            in_channels,
            n_classes,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            p=p,
            mlp_ratio=mlp_ratio,
        )

    if network == "hybridvit":
        img_size = (224, input_dim)
        patch_size = 224  # [32,32]  [16,16]
        in_channels = 1
        num_classes = 6
        dim = 512  # 384 # 512 #2048
        depth = 3  # 6  4
        n_heads = 4
        mlp_dim = 512
        dropout = 0.1
        emb_dropout = 0.1
        n_filter_list = [1, 16, 32, 64]
        seq_pool = False
        positional_embedding = True

        model = HybridViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=n_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
            dropout=dropout,
            n_filter_list=n_filter_list,
            emb_dropout=emb_dropout,
            seq_pool=seq_pool,
            positional_embedding=positional_embedding,
        )

    if network == "powerhybridvit":
        set_power_prop_global(1.5)
        img_size = (224, input_dim)
        patch_size = 224  # [32,32]  [16,16]
        in_channels = 1
        num_classes = 6
        dim = 512  # 384 # 512 #2048
        depth = 3  # 6  4
        n_heads = 4
        mlp_dim = 512
        dropout = 0.1
        emb_dropout = 0.1
        n_filter_list = [1, 16, 32, 64]
        seq_pool = False
        positional_embedding = True

        model = HybridViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=n_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
            dropout=dropout,
            n_filter_list=n_filter_list,
            emb_dropout=emb_dropout,
            seq_pool=seq_pool,
            positional_embedding=positional_embedding,
        )

    if network == "resnet":
        model = ResNet(34, Block, image_channels=7, num_classes=6)

    if network == "resnet_custom":
        model = ResNet(34, Block, image_channels=1, num_classes=6)

    if network == "toy":
        model = toy_model_generator(input_channels=1, output_dim=6)()

    return model


# Define metric aggregation function
def get_generate_model(network, input_dim=1120):
    def get_model():
        nonlocal network
        nonlocal input_dim
        return get_network(network=network, input_dim=input_dim)

    def extract_federated_model(input):
        return input

    return get_model, extract_federated_model


def get_generate_local_federated_models(
    federated_network, local_network, input_dim=1120
):
    def get_fed_model():
        return get_network(network=federated_network, input_dim=input_dim)

    def get_local_model():
        return get_network(network=local_network, input_dim=input_dim)

    def extract_federated_model(input):
        return input[0]

    return (get_fed_model, get_local_model), extract_federated_model


def gen_on_fit_config_fn(
    epochs_per_round: int,
    batch_size: int,
    client_learning_rate: float,
    weight_decay,
    gamma,
    step_size,
    num_workers: int,
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": server_round,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "step_size": step_size,
            "num_workers": num_workers,
        }
        return local_config

    return on_fit_config


def gen_mixed_async_on_fit_config_fn(
    replay_epochs: int,
    async_epochs: int,
    batch_size: int,
    client_learning_rate: float,
    weight_decay: float,
    gamma: float,
    step_size: int,
    num_workers: int,
    train_inc: int,
    replay_store_max_batches: int,
    federated_alpha: float,
    local_alpha: float,
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """
    train_start_index = 0
    train_inc = train_inc * batch_size
    replay_store_max_batches = replay_store_max_batches * batch_size

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        nonlocal train_start_index, train_inc
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": server_round,
            "replay_epochs": replay_epochs,
            "async_epochs": async_epochs,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
            "weight_decay": weight_decay,
            "gamma": gamma,
            "step_size": step_size,
            "num_workers": num_workers,
            "train_start_index": train_start_index,
            "train_inc": train_inc,
            "replay_store_max_size": replay_store_max_batches,
            "federated_alpha": federated_alpha,
            "local_alpha": local_alpha,
        }
        train_start_index += train_inc

        return local_config

    return on_fit_config


def gen_on_evaluate_config_fn(
    batch_size: int,
    num_workers: int,
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` on_evaluate_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_evaluate_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
        return local_config

    return on_evaluate_config


# Federated evaluation function
def get_fed_eval_fn(
    fed_test_file: PathLike,
    fed_test_sampling_file: PathLike,
    batch_size: int,
    num_workers: int,
    transform_x,
    transform_y,
    model_generator: Callable[[], torch.nn.Module],
    criterion_generator: Callable[[], torch.nn.Module] = torch.nn.CrossEntropyLoss,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Returns a federated evaluation function to be run on the server

    Args:
        fed_dir (PathLike): File for the federated test set
        model_generator (Callable[[], torch.nn.Module]): generates a model of the specified type

    Returns:
        Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]: Function which evaluates the federated model performance
    """
    fed_test_file = Path(fed_test_file)
    fed_test_sampling_file = Path(fed_test_sampling_file)

    X, y = torch.load(fed_test_file)
    sampling = torch.load(fed_test_sampling_file)

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nonlocal X
        nonlocal y
        nonlocal sampling
        nonlocal batch_size
        nonlocal num_workers
        nonlocal transform_x
        nonlocal transform_y

        # pylint: disable=unused-argument
        """Use the federated test set for evaluation"""
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model_generator()
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = prepare_one_dataloader(
            X_train=X,
            y_train=y,
            sampling=sampling,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_x=transform_x,
            transform_y=transform_y,
        )
        loss, accuracy, f1_macro = test_client(
            net=net,
            testloader=testloader,
            device=device,
            criterion=criterion_generator(),
        )  # type:ignore
        # return statistics
        return loss, {"accuracy": accuracy, "f1_macro": f1_macro}

    return evaluate


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
    output_directory: Path,
) -> None:
    """Simple plotting method for Classification Task.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.savefig(output_directory / "graph.png")
    plt.close()


def get_initial_parameters(
    model_generator: Callable, resume_model: Optional[str]
) -> Parameters:
    """Generates a model and returns its parameters to be used for initialization

    Args:
        model_generator (Callable): Function generating a model

    Returns:
        Parameters: Parameters to be sent back to the server for initializing the model
    """
    if resume_model is None:
        weights = [
            val.cpu().numpy() for _, val in model_generator().state_dict().items()
        ]
    else:
        weights = [
            arr
            for val in np.load(resume_model, allow_pickle=True).values()
            for arr in val
        ]
    parameters = ndarrays_to_parameters(weights)

    return parameters


@wrapper_func
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    average_dict: Dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }
