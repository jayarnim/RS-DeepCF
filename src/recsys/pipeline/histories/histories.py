from torch.nn.utils.rnn import pad_sequence


def histories_generator(
    selector,
    padding_idx: int,
):
    # select hist per anchor
    kwargs = dict()
    hist_indices = selector(**kwargs)

    # padding
    kwargs = dict(
        sequences=hist_indices, 
        batch_first=True, 
        padding_value=padding_idx,
    )
    hist_indices_padded = pad_sequence(**kwargs)

    return hist_indices_padded