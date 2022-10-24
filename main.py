"""Runs AdaptiveFederated Optimization for CIFAR10/100."""
from os import chdir, getcwd, environ

from pathlib import Path

import flwr as fl
import hydra
from flwr.common.typing import Parameters
from flwr.server import ServerConfig
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from flwr.common.parameter import ndarrays_to_parameters
import json
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import shutup

shutup.please()


@hydra.main(
    config_path="conf/federated/default", config_name="config", version_base=None
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

    # Just to test before launching ray
    model = model_generator()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]

    src_dir = Path(original_cwd)
    fed_dir = Path(cfg.fed_dir)

    # Create federated partitions - checkout the config files for details
    # path_original_dataset = Path(to_absolute_path(cfg.root_dir))
    if cfg.recreate_partitions:
        fed_dir.mkdir(parents=True, exist_ok=True)
        call(cfg.generate_partitions, fed_dir=fed_dir)

    # Define client resources and ray configs
    client_resources = {
        "num_cpus": cfg.cpus_per_client,
        "num_gpus": cfg.gpus_per_client,
    }
    ray_config = {"include_dashboard": cfg.ray_config.include_dashboard}

    on_fit_config_fn = call(
        cfg.gen_on_fit_config_fn, client_learning_rate=cfg.strategy.eta_l
    )

    on_evaluate_config_fn = call(cfg.gen_on_evaluate_config_fn)

    initial_parameters = call(
        cfg.get_initial_parameters, model_generator=model_generator
    )

    # Get centralized evaluation function - see config files for details
    evaluate_fn = call(
        cfg.get_fed_eval_fn,
        model_generator=model_generator,
    )

    fit_agg_func = call(cfg.get_on_fit_metrics_agg_fn)
    eval_agg_func = call(cfg.get_on_evaluate_metrics_agg_fn)

    strategy = instantiate(
        cfg.strategy.init,
        fraction_fit=(float(cfg.num_clients_per_round) / cfg.num_total_clients),
        fraction_evaluate=(
            float(cfg.num_evaluate_clients_per_round) / cfg.num_total_clients
        ),
        min_fit_clients=cfg.num_clients_per_round,
        min_evaluate_clients=cfg.num_evaluate_clients_per_round,
        min_available_clients=cfg.num_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        accept_failures=False,
        fit_metrics_aggregation_fn=fit_agg_func,
        evaluate_metrics_aggregation_fn=eval_agg_func,
        save_rounds=cfg.save_rounds,
        save_directory=output_directory,
    )

    strategy.initial_parameters = initial_parameters

    # start simulation
    if cfg.is_simulation:
        client_fn = call(
            cfg.get_ray_client_fn,
            model_generator=model_generator,
            fed_dir=fed_dir,
        )
        client = client_fn(0)
        print(client.fit(weights, on_fit_config_fn(0))[-1])
        print(client.evaluate(weights, on_evaluate_config_fn(0)))
        evaluate_fn(0, weights, on_evaluate_config_fn(0))
        hist = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_total_clients,
            client_resources=client_resources,
            config=ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            ray_init_args=ray_config,
        )
    else:  # or start server
        hist = fl.server.app.start_server(
            server_address=cfg.server_address,
            server=cfg.server,
            config=ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
        )

    # Plot results
    call(cfg.plot_results, hist=hist, output_directory=output_directory)
    with open(output_directory / "hist.json", "w", encoding="utf-8") as f:
        json.dump(hist.__dict__, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
