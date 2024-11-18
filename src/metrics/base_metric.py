from abc import abstractmethod
from torchmetrics.audio import PermutationInvariantTraining


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, use_pit=False, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__
        if use_pit:
            self.metric = PermutationInvariantTraining(self.metric).to('cuda')

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()
