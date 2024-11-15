import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import setup_saving_and_logging
from deepspeed.profiling.flops_profiler import FlopsProfiler
from src.trainer import Inferencer
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


def measure_model_params(model):
    model.train()
    return {'total_params_count': sum(p.numel() for p in model.parameters()),
            'trainable_params_count': sum(p.numel() for p in model.parameters() if p.requires_grad)}


@hydra.main(
    version_base=None, config_path="src/configs", config_name="calculate_bonus_train"
)
def measure_train(config):
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    profiler = FlopsProfiler(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
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

    print()
    for key, value in trainer.profiler_data.items():
        if 'macs' in key or 'flops' in key:
            print(key, value / 10**9)
        elif 'params' in key:
            print(key, value / 10**6)
        else:
            print(key, value)
    print()


@hydra.main(
    version_base=None, config_path="src/configs", config_name="calculate_bonus_inference"
)
def measure_inference(config):
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    separate_only = dataloaders["test"].dataset.separate_only

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    profiler = FlopsProfiler(model)

    # get metrics
    metrics = instantiate(config.metrics) if not separate_only else None

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=True,
        separate_only=separate_only,
        profiler=profiler,
    )
    inferencer.run_inference()

    print()
    for key, value in inferencer.profiler_data.items():
        if 'macs' in key or 'flops' in key:
            print(key, value / 10**9)
        elif 'params' in key:
            print(key, value / 10**6)
        else:
            print(key, value)
    print()
    for key, value in measure_model_params(model).items():
        print(key, value / 10**6)
    print()


if __name__ == "__main__":
    measure_train()
    measure_inference()