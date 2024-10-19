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
        "source": torch.zeros((batch_size, 2, time_audio)),
        "video": torch.zeros((batch_size, 2, time_video, h, w)),
        "source1_path": [""] * batch_size,
        "source2_path": [""] * batch_size,
    }
    for i in range(batch_size):
        result_batch["mix"][i] = dataset_items[i]["mix"]
        result_batch["source"][i][0] = dataset_items[i]["label1"]
        result_batch["source"][i][1] = dataset_items[i]["label2"]
        result_batch["video"][i][0] = dataset_items[i]["mouths1"]
        result_batch["video"][i][1] = dataset_items[i]["mouths2"]
        result_batch["source1_path"][i] = dataset_items[i]["label1_path"]
        result_batch["source2_path"][i] = dataset_items[i]["label2_path"]
    return result_batch
