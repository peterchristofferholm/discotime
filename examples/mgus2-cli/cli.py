from warnings import filterwarnings

from lightning.pytorch import cli
from lightning.pytorch.callbacks import EarlyStopping


class CustomLightningCLI(cli.LightningCLI):
    def add_arguments_to_parser(
        self, parser: cli.LightningArgumentParser
    ) -> None:
        parser.add_lightning_class_args(
            EarlyStopping, "callbacks.early_stopping"
        )


def cli_main(args: cli.ArgsType = None):
    from discotime.datasets import Mgus2
    from discotime.models import LitSurvModule

    filterwarnings("ignore", message=".*`num_workers`")
    CustomLightningCLI(
        model_class=LitSurvModule,
        datamodule_class=Mgus2,
        save_config_callback=None,
        seed_everything_default=123,
        args=args,
    )


if __name__ == "__main__":
    cli_main()
