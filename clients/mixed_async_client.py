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
from data.partition_data import get_partition_data, prepare_one_dataloader
from train import train_client, test_client, train_federated_local_client
from utils.flower import save_model
from flwr.common.parameter import parameters_to_ndarrays, ndarrays_to_parameters
from datetime import datetime
import base64
import os


class MixedAsyncRayClient(fl.client.NumPyClient):
    """Ray Virtual Client."""

    def __init__(
        self,
        cid: str,
        fed_dir: Path,
        model_generator: Callable,
        criterion_generator: Callable,
        optimizer_generator: Callable,
        scheduler_generator: Callable,
        transform_x: Callable,
        transform_y: Callable,
        unique_experiment_id,
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
        self.model_generator, self.local_model_generator = model_generator
        self.fed_dir = fed_dir
        self.criterion_generator = criterion_generator
        self.optimizer_generator = optimizer_generator
        self.scheduler_generator = scheduler_generator
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.unique_experiment_id = unique_experiment_id

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

    def load_model_and_replay_store(self, save_dir, ext="pt"):
        save_dir = Path(save_dir)
        full_path = save_dir / f"local_model_and_data_{self.unique_experiment_id}.{ext}"
        if full_path.exists():
            weights = [
                arr
                for val in np.load(full_path, allow_pickle=True).values()
                for arr in val
            ]
            parameters = ndarrays_to_parameters(weights)
            return parameters
        else:
            weights = [
                val.cpu().numpy()
                for _, val in self.local_model_generator().state_dict().items()
            ]
            replay_store = []
            weights.append(replay_store)
            return weights

    def save_model_and_replay_store(self, local_weights, ext="pt"):
        save_model(
            params=local_weights,
            save_dir=Path(self.fed_dir) / {self.cid},
            save_name=f"local_model_and_data_{self.unique_experiment_id}",
            ext=ext,
        )

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
        unique_experiment_id = config["unique_experiment_id"]

        local_data = self.load_model_and_replay_store(
            save_dir=Path(self.fed_dir) / {self.cid},
        )

        replay_store = [val for val in local_data.pop()]

        replay_store_max_size = config["replay_store_max_size"]

        train_start_index = config["train_start_index"]
        train_inc = config["train_inc"]

        federated_model = self.set_federated_parameters(parameters=parameters)
        federated_model.to(self.device)

        local_model = self.set_local_parameters(local_data)

        # train
        X, y, sampling = get_partition_data(
            path_to_data=Path(self.fed_dir),
            cid=self.cid,
            partition_type="train",
        )

        if train_start_index + train_inc > len(y):
            if train_start_index + train_inc >= len(y) + train_inc:
                train_start_index = np.random.randint(0, max(len(y) - train_inc, 0))
            else:
                train_inc = len(y) - train_start_index

        replay_loader = prepare_one_dataloader(
            X_train=X,
            y_train=y,
            sampling=sampling,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            transform_x=self.transform_x,
            transform_y=self.transform_y,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(replay_store),
        )

        async_loader = prepare_one_dataloader(
            X_train=X,
            y_train=y,
            sampling=sampling,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            transform_x=self.transform_x,
            transform_y=self.transform_y,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                list(range(train_start_index, train_start_index + train_inc))
            ),
        )

        federated_optimizer = self.optimizer_generator(
            federated_model.parameters(),
            lr=config["client_learning_rate"],
            weight_decay=0.01,
        )

        local_optimizer = self.optimizer_generator(
            local_model.parameters(),
            lr=config["client_learning_rate"],
            weight_decay=0.01,
        )

        scheduler = self.scheduler_generator(
            federated_optimizer, step_size=config["step_size"], gamma=config["gamma"]
        )
        train_federated_local_client(
            federated_model=federated_model,
            local_model=local_model,
            federated_optimizer=federated_optimizer,
            local_optimizer=local_optimizer,
            federated_alpha=config["federated_alpha"],
            local_alpha=config["local_alpha"],
            trainloader=replay_loader,
            epochs=config["replay_epochs"],
            device=self.device,
            criterion=self.criterion_generator(),
        )
        train_client(
            local_model,
            trainloader=async_loader,
            epochs=config["async_epochs"],
            device=self.device,
            optimizer=local_optimizer,
            criterion=self.criterion_generator(),
        )

        # return local model and statistics
        federated_weights = [
            val.cpu().numpy() for _, val in federated_model.state_dict().items()
        ]

        replay_store.extend(
            list(range(train_start_index, train_start_index + train_inc))
        )

        if len(replay_store) > replay_store_max_size:
            new_start = len(replay_store) - replay_store_max_size
            new_replay_store = replay_store[new_start:]

        local_weights = [
            val.cpu().numpy() for _, val in local_model.state_dict().items()
        ]

        local_weights.append(np.array(new_replay_store))

        self.save_model_and_replay_store(
            local_weights=local_weights,
        )

        return federated_weights, len(replay_loader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Implements di\stributed evaluation for a given client.

        Args:
            parameters (NDArrays): Set of weights being used for evaluation
            config (Dict[str, Scalar]): Dictionary containing possible options for
            evaluations.

        Returns:
            Tuple[float, int, Dict[str, float]]: Loss, number of samples and dictionary
            of metrics.
        """
        federated_model = self.set_parameters(parameters)
        federated_model.to(self.device)

        *local_data, _ = self.load_model_and_replay_store(
            save_dir=Path(self.fed_dir) / {self.cid},
        )

        local_model = self.set_local_parameters(local_data)
        local_model.to(self.device)

        # load data for this client and get valloader
        X, y, sampling = get_partition_data(
            path_to_data=Path(self.fed_dir),
            cid=self.cid,
            partition_type="test",
        )

        valid_loader = prepare_one_dataloader(
            X_train=X,
            y_train=y,
            sampling=sampling,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            transform_x=self.transform_x,
            transform_y=self.transform_y,
        )
        # evaluate
        federated_loss, federated_accuracy, federated_f1 = test_client(
            net=federated_model,
            testloader=valid_loader,
            device=self.device,
            criterion=self.criterion_generator(),
        )

        local_model_loss, local_model_acc, local_model_f1 = test_client(
            net=local_model,
            testloader=valid_loader,
            device=self.device,
            criterion=self.criterion_generator(),
        )
        # return statistics
        return (
            float(federated_loss),
            len(valid_loader.dataset),
            {
                "eval_global_accuracy": float(federated_accuracy),
                "eval_global_loss": float(federated_loss),
                "eval_global_f1_macro": federated_f1,
                "eval_local_accuracy": float(local_model_acc),
                "eval_local_loss": float(local_model_loss),
                "eval_local_f1": local_model_f1,
            },
        )  # type: ignore

    def set_parameters(self, net, parameters: NDArrays):
        """Loads weights inside the network.

        Args:
            parameters (NDArrays): set of weights to be loaded.

        Returns:
            [type]: Network with new set of weights.
        """
        weights = parameters
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net

    def set_federated_parameters(self, parameters: NDArrays):
        net = self.model_generator()
        return self.set_parameters(net=net, parameters=parameters)

    def set_local_parameters(self, parameters: NDArrays):
        net = self.local_model_generator()
        return self.set_parameters(net=net, parameters=parameters)


def get_mixed_async_client_fn(
    model_generator,
    fed_dir,
    transform_x: Callable,
    transform_y: Callable,
    criterion_generator: Callable = nn.CrossEntropyLoss,
    optimizer_generator: Callable = torch.optim.AdamW,
    scheduler_generator: Callable = torch.optim.lr_scheduler.StepLR,
) -> Callable[[str], MixedAsyncRayClient]:

    """Function that loads a Ray (Virtual) Client.

    Args:
        fed_dir (Path): Path containing local datasets in the form ./client_id/train.pt
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Callable[[str], RayClient]: [description]
    """

    unique_experiment_id = base64.b64encode(os.urandom(32))[:8]

    def client_fn(cid: str) -> MixedAsyncRayClient:

        # create a single client instance
        return MixedAsyncRayClient(
            cid=cid,
            fed_dir=fed_dir,
            model_generator=model_generator,
            criterion_generator=criterion_generator,
            optimizer_generator=optimizer_generator,
            scheduler_generator=scheduler_generator,
            transform_x=transform_x,
            transform_y=transform_y,
            unique_experiment_id=unique_experiment_id,
        )

    return client_fn
