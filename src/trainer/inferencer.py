import torch
import torchaudio
from torch import autocast
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        separate_only=True,
        profiler=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (Path): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.profiler = profiler
        self.profiler_data = {}
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.is_amp = config.inferencer.get("is_amp", True)

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))
        self.separate_only = separate_only

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        if self.profiler is not None:
            self.profiler.start_profile()
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        if self.profiler is not None:
            self.profiler.stop_profile()
            self.profiler_data['inference_flops'] = self.profiler.get_total_flops()
            self.profiler_data['inference_macs'] = self.profiler.get_total_params()
            self.profiler_data['inference_params'] = self.profiler.get_total_params()
            self.profiler_data['inference_time'] = self.profiler.get_total_duration()
            self.profiler.end_profile()
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        with autocast(
            device_type=self.device, enabled=self.is_amp, dtype=torch.float16
        ):
            outputs = self.model(**batch)
            outputs["predict"] *= 200
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = batch["predict"].shape[0]
            for i in range(batch_size):
                mix = batch["mix"][i].clone()
                predict = batch["predict"][i].clone()

                torchaudio.save(
                    self.save_path / "mix" / batch["name"][i],
                    mix.unsqueeze(0).cpu(),
                    16000,
                )
                torchaudio.save(
                    self.save_path / "s1" / batch["name"][i],
                    predict[0].unsqueeze(0).cpu(),
                    16000,
                )
                torchaudio.save(
                    self.save_path / "s2" / batch["name"][i],
                    predict[1].unsqueeze(0).cpu(),
                    16000,
                )
        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if not self.separate_only:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / "mix").mkdir(exist_ok=True, parents=True)
            (self.save_path / "s1").mkdir(exist_ok=True, parents=True)
            (self.save_path / "s2").mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result() if not self.separate_only else None

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if batch[tensor_for_device] is None:
                continue
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch
