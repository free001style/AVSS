import os
import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate

from src.datasets.collate import collate_fn_metrics
from src.metrics.tracker import MetricTracker

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(
    version_base=None, config_path="src/configs", config_name="calculate_metrics"
)
def main(config):
    metrics = instantiate(config.metrics)["inference"]
    evaluation_metrics = MetricTracker(
        *[m.name for m in metrics],
        writer=None,
    )
    predict_dir = Path(config.predict_dir)
    gt_dir = Path(config.gt_dir)
    dataset = instantiate(config.datasets, predict_dir, gt_dir)
    dataloader = instantiate(
        config.dataloader, dataset=dataset, collate_fn=collate_fn_metrics
    )
    for batch in dataloader:
        print(batch["source"].shape)
        for met in metrics:
            evaluation_metrics.update(met.name, met(**batch))
    logs = evaluation_metrics.result()
    for key, value in logs.items():
        print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()
