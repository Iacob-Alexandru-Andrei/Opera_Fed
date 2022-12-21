"""Runs AdaptiveFederated Optimization for CIFAR10/100."""
from collections import defaultdict
from email.policy import default
from os import chdir, getcwd, environ
from pathlib import Path

import flwr as fl
import hydra
from flwr.common.typing import Parameters
from flwr.server import ServerConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
import json
import warnings
from collections import defaultdict
from typing import List, Dict, Any

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import shutup

shutup.please()


@hydra.main(
    config_path="conf/federated/evaluate/default",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """General-purpose main function that receives cfg from Hydra."""
    # Make sure we are on the right directory.
    # This will not be necessary in hydra 1.3
    environ["HYDRA_FULL_ERROR"] = "1"
    environ["NUMEXPR_MAX_THREADS"] = "12"
    output_directory = Path(to_absolute_path(HydraConfig.get().runtime.output_dir))
    original_cwd = get_original_cwd()
    chdir(original_cwd)

    model_generator = call(cfg.get_generate_model)
    transform_x = call(cfg.get_transform_x)
    transform_y = call(cfg.get_transform_y)

    initial_parameters = parameters_to_ndarrays(
        call(cfg.get_initial_parameters, model_generator=model_generator)
    )

    results: List[Dict[str, Any]] = []
    for model, fed_test_files, sampling_files in zip(
        cfg.models, cfg.fed_test_files, cfg.fed_test_sampling_files
    ):
        results_dict: Dict[str, Any] = {
            "model": cfg.resume_model,
            "data": [],
        }
        for fed_test_file, sampling_file in zip(fed_test_files, sampling_files):
            eval_fn = call(
                cfg.get_fed_eval_fn,
                model_generator=model_generator,
                fed_test_file=fed_test_file,
                fed_test_sampling_file=sampling_file,
                transform_x=transform_x,
                transform_y=transform_y,
            )
            loss, met = eval_fn(
                server_round=0, parameters_ndarrays=initial_parameters, config={}
            )
            results_exp = {
                "loss": loss,
                "metrics": met,
                "test_file": fed_test_files,
                "sampling_file": sampling_files,
            }
            results_dict["data"].append(results_dict)
        results.append(results_exp)

    with open(output_directory / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
