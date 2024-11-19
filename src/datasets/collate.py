import torch


def collate_fn(dataset_items: list[dict], use_video):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
        use_video (bool): whether to add video key in dict.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch_size = len(dataset_items)
    time_audio = dataset_items[0]["mix"].shape[0]
    result_batch = {
        "mix": torch.zeros((batch_size, time_audio)),
        "source": torch.zeros((batch_size, 2, time_audio))
        if dataset_items[0]["label1"] is not None
        else None,
        "video": torch.zeros((batch_size, 2, *dataset_items[0]["mouths1"].shape))
        if use_video
        else None,
        "name": [""] * batch_size,
    }
    for i in range(batch_size):
        result_batch["mix"][i] = dataset_items[i]["mix"]
        if result_batch["source"] is not None:
            result_batch["source"][i][0] = dataset_items[i]["label1"]
            result_batch["source"][i][1] = dataset_items[i]["label2"]
        if result_batch["video"] is not None:
            result_batch["video"][i][0] = dataset_items[i]["mouths1"]
            result_batch["video"][i][1] = dataset_items[i]["mouths2"]
        result_batch["name"][i] = dataset_items[i]["name"]
    return result_batch


def collate_fn_metrics(dataset_items: list[dict]):
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
    result_batch = {
        "mix": torch.zeros((batch_size, time_audio)),
        "source": torch.zeros((batch_size, 2, time_audio)),
        "predict": torch.zeros((batch_size, 2, time_audio)),
    }
    for i in range(batch_size):
        result_batch["mix"][i] = dataset_items[i]["mix"]
        result_batch["source"][i] = dataset_items[i]["source"]
        result_batch["predict"][i] = dataset_items[i]["predict"]
    return result_batch
