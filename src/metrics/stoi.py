from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies pesq metric function

        """
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(16000)

    def __call__(self, source, predict, **kwargs):
        """
        Metric calculation logic.

        Args:
            source (Tensor): ground-truth waveforms.
            predict (Tensor): predicted waveforms.

        Returns:
            si-snri (float): calculated si-snri.
        """
        if source.size() != predict.size() or source.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {source.size()} and {predict.size()} instead"
            )

        return self.metric(predict, source).mean()
