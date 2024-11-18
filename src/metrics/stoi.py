from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies STOI metric function.
        """
        self.metric = ShortTimeObjectiveIntelligibility(16000)
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
        Returns:
            metric (Tensor): calculated STOI.
        """
        return self.metric(predict, source).mean()
