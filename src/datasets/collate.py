import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)
    time_audio = dataset_items[0]["mix"].shape[0]
    time_video, h, w = dataset_items[0]["mouths1"].shape
    result_batch = {
        "mix": torch.zeros((batch_size, time_audio)),
        "label1": torch.zeros((batch_size, time_audio)),
        "label2": torch.zeros((batch_size, time_audio)),
        "mouths1": torch.zeros((batch_size, time_video, h, w)),
        "mouths2": torch.zeros((batch_size, time_video, h, w)),
        "label1_path": [""] * batch_size,
        "label2_path": [""] * batch_size,
    }
    for i in range(batch_size):
        for key in result_batch.keys():
            result_batch[key][i] = dataset_items[i][key]
    return result_batch
