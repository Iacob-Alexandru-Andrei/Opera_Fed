"""Flower Client for CIFAR10/100."""
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple
import torch.nn as nn
import torch

import flwr as fl
import numpy as np

from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader
from data.partition_data import get_partition_data
from train import train_client, test_client


class RayClient(fl.client.NumPyClient):
    """Ray Virtual Client."""

    def __init__(
        self,
        cid: str,
        fed_dir: Path,
        model_generator: Callable,
        criterion_generator: Callable,
        optimizer_generator: Callable,
        scheduler_generator: Callable,
    ):
        """Implements Ray Virtual Client.

        Args:
            cid (str): Client ID, in our case a str representation of an int.
            fed_dir (Path): Path where partitions are saved.
            num_classes (int): Number of classes in the classification problem.
        """
        self.cid = cid
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_generator = model_generator
        self.fed_dir = fed_dir
        self.criterion_generator = criterion_generator
        self.optimizer_generator = optimizer_generator
        self.scheduler_generator = scheduler_generator

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Returns properties for this client.

        Args:
            config (Dict[str, Scalar]): Options to be used for selecting specific
            properties.

        Returns:
            Dict[str, Scalar]: Returned properties.
        """
        # pylint: disable=unused-argument
        return self.properties

    def get_parameters(self, config) -> NDArrays:
        """Returns weight from a given model. If no model is passed, then a
        local model is created. This can be used to initialize a model in the
        server.

        Returns:
            NDArrays: weights from the model.
        """
        net = self.model_generator()
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Usual fit function that performs training locally.

        Args:
            parameters (NDArrays): Initial set of weights sent by the server.
            config (Dict[str, Scalar]): config file containing num_epochs,etc...

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: New set of weights,
            number of samples and dictionary of metrics.
        """
        net = self.set_parameters(parameters)
        net.to(self.device)
        # train
        train_loader = get_partition_data(
            path_to_data=Path(self.fed_dir),
            cid=self.cid,
            partition_type="train",
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        optimizer = self.optimizer_generator(
            net.parameters(), lr=config["client_learning_rate"], weight_decay=0.01
        )
        scheduler = self.scheduler_generator(
            optimizer, step_size=config["step_size"], gamma=config["gamma"]
        )
        loss = train_client(
            net,
            trainloader=train_loader,
            epochs=config["epochs"],
            device=self.device,
            optimizer=optimizer,
            criterion=self.criterion_generator(),
        )
        # return local model and statistics
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights, len(train_loader), {"loss": loss}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Implements distributed evaluation for a given client.

        Args:
            parameters (NDArrays): Set of weights being used for evaluation
            config (Dict[str, Scalar]): Dictionary containing possible options for
            evaluations.

        Returns:
            Tuple[float, int, Dict[str, float]]: Loss, number of samples and dictionary
            of metrics.
        """
        net = self.set_parameters(parameters)
        net.to(self.device)
        # load data for this client and get valloader
        valid_loader = get_partition_data(
            path_to_data=Path(self.fed_dir),
            cid=self.cid,
            partition_type="test",
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        # evaluate
        loss, accuracy = test_client(
            net=net,
            testloader=valid_loader,
            device=self.device,
            criterion=self.criterion_generator(),
        )
        # return statistics
        return float(loss), len(valid_loader.dataset), {"accuracy": float(accuracy)}  # type: ignore

    def set_parameters(self, parameters: NDArrays):
        """Loads weights inside the network.

        Args:
            parameters (NDArrays): set of weights to be loaded.

        Returns:
            [type]: Network with new set of weights.
        """
        net = self.model_generator()
        weights = parameters
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net


def get_ray_client_fn(
    model_generator,
    fed_dir,
    criterion_generator: Callable = nn.CrossEntropyLoss,
    optimizer_generator: Callable = torch.optim.AdamW,
    scheduler_generator: Callable = torch.optim.lr_scheduler.StepLR,
) -> Callable[[str], RayClient]:

    """Function that loads a Ray (Virtual) Client.

    Args:
        fed_dir (Path): Path containing local datasets in the form ./client_id/train.pt
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Callable[[str], RayClient]: [description]
    """

    def client_fn(cid: str) -> RayClient:

        # create a single client instance
        return RayClient(
            cid=cid,
            fed_dir=fed_dir,
            model_generator=model_generator,
            criterion_generator=criterion_generator,
            optimizer_generator=optimizer_generator,
            scheduler_generator=scheduler_generator,
        )

    return client_fn
