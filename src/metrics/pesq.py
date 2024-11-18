from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Applies PESQ metric function.
        """
        self.metric = PerceptualEvaluationSpeechQuality(16000, "wb")
        super().__init__(*args, **kwargs)

    def __call__(self, source, predict, **kwargs):
        """
        Args:
            source (Tensor): (B, n_spk, T) ground-truth speech.
            predict (Tensor): (B, n_spk, T) predicted speech.
        Returns:
            metric (Tensor): calculated PESQ.
        """
        return self.metric(predict, source).mean()
