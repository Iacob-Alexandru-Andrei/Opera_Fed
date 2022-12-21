from flwr.server.strategy import FedAdagrad, FedAvg, FedAdam, FedAvgM, FedYogi, Strategy
import torch
from typing import List
import numpy as np
from flwr.common.parameter import parameters_to_ndarrays
from pathlib import Path


class SaveMixin:
    def __init__(self, *args, **kwargs):
        save_rounds = kwargs.pop("save_rounds")
        save_directory = kwargs.pop("save_directory")
        super().__init__(*args, **kwargs)
        self.save_rounds = save_rounds
        self.save_directory = save_directory

    def aggregate_fit(self, *args, **kwargs):
        server_round = args[0]
        params, metrics = super().aggregate_fit(*args, **kwargs)
        if params is not None and server_round in self.save_rounds:
            aggregate_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
            save_dir = Path(self.save_directory) / "saved_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_dir / f"round_{server_round}_weights.pt",
                aggregate_ndarrays,
            )
        return params, metrics


class SaveFedAvg(SaveMixin, FedAvg):
    pass


class SaveFedAdagrad(SaveMixin, FedAdagrad):
    pass


class SaveFedAdamd(SaveMixin, FedAdam):
    pass


class SaveFedAvgM(SaveMixin, FedAvgM):
    pass


class SaveFedYogi(SaveMixin, FedYogi):
    pass
