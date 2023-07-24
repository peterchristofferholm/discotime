import logging
import yaml
from collections.abc import Mapping

import optuna
from optuna.trial import Trial

from gse5479_cli import cli_main
from discotime.utils.misc import (
    get_last_intermediate_value,
    update_mapping,
    recursive_defaultdict,
    OptunaCallback,
)

###############################################################################

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

###############################################################################


def get_config(trial: Trial) -> Mapping:
    hparams_cfg = recursive_defaultdict()
    hparams_cfg["model"]["config"] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-1),
        "activation_function": trial.suggest_categorical(
            "activation_function", ["ReLU", "SiLU", "GELU", "Mish"]
        ),
        "n_sequential_blocks": trial.suggest_int("n_sequential_blocks", 1, 30),
        "n_hidden_units": trial.suggest_int("n_hidden_units", 1, 50),
        "dropout_rate": trial.suggest_float("droprate", 0, 0.8),
        "batch_normalization": trial.suggest_categorical(
            "batch_normalization", [False, True]
        ),
        "use_skip_connections": trial.suggest_categorical(
            "use_skip_connections", [False, True]
        ),
    }
    hparams_cfg["data"]["data_config"] = {
        "batch_size": trial.suggest_int("batch_size", 2, 64),
        "n_time_bins": trial.suggest_int("n_time_bins", 1, 40),
    }
    hparams_cfg["callbacks"]["early_stopping"] = {
        "patience": trial.suggest_int("patience", 1, 50)
    }

    with open("default_config.yaml") as f:
        default_cfg = yaml.safe_load(f)

    config = update_mapping(default_cfg, hparams_cfg)

    return config


def objective(trial: Trial) -> float:
    config = get_config(trial)
    config["trainer"]["callbacks"] = [
        OptunaCallback(trial, "val_IPA_cause1", minvalue=-1)
    ]

    # launch the trainer through the CLI
    cli_main(args={"fit": config})

    # save config in optuna storage
    config["trainer"]["callbacks"] = None
    trial.set_user_attr("config", config)

    return get_last_intermediate_value(trial)


def main():
    from optuna.storages import JournalStorage, JournalFileStorage

    storage = JournalStorage(JournalFileStorage("./optuna.log"))
    study = optuna.create_study(
        study_name="gse5479",
        direction="maximize",
        storage=storage,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=5),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=500, gc_after_trial=True)

    with open("best_config.yaml", "wt") as f:
        config = get_config(study.best_trial)
        config["trainer"]["enable_checkpointing"] = True
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
