import os
import sys
import warnings

sys.path.append('.')
import hydra
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer, Trainer
from src.utils.init_utils import setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(
    version_base=None, config_path="../src/configs", config_name="calculate_efficiency"
)
def measure(config):
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    profiler = FlopsProfiler(model)
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        profiler=profiler,
    )

    trainer.train()

    print("All Params (M)", sum([p.numel() for p in model.parameters()]) / 10 ** 6)
    print("Trainable Params (M)", sum([p.numel() for p in model.parameters() if p.requires_grad]) / 10 ** 6)
    print("MACs (G)", trainer.profiler_data["macs"] / 10 ** 9)
    print("Memory (GB)", trainer.profiler_data["train_memory"])
    print("Train time (s)", trainer.profiler_data["train_time"])
    print("Infer. time (s)", trainer.profiler_data["eval_time"])
    print("Real-Time factor", trainer.profiler_data["eval_time"] / 2)
    print("Size of the saved model on disk (MB)", os.path.getsize(config.model_path) / 10 ** 6)


if __name__ == "__main__":
    measure()
