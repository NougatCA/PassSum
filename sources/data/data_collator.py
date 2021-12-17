
import torch

from typing import List
import itertools

from data.vocab import Vocab
import enums


def collate_fn(batch, args, code_vocab, node_vocab):
    """
    Data collator function.
    """
    model_inputs = {}

    code_raw, path_raw, nl_raw = map(list, zip(*batch))

    model_inputs['code_input_ids'], model_inputs['attention_mask'] = get_batch_inputs(
        batch=code_raw,
        vocab=code_vocab,
        processor=Vocab.sep_processor,
        max_len=args.max_code_len
    )

    model_inputs['id_inputs'], model_inputs['id_seq_lens'], model_inputs['node_inputs'], model_inputs['node_seq_lens']

    model_inputs['decoder_input_ids'], model_inputs['decoder_attention_mask'] = get_batch_inputs(
        batch=nl_raw,
        vocab=code_vocab,
        processor=Vocab.sos_processor,
        max_len=args.max_nl_len,
    )
    model_inputs['labels'], _ = get_batch_inputs(
        batch=nl_raw,
        vocab=code_vocab,
        processor=Vocab.eos_processor,
        max_len=args.max_nl_len,
    )

    return model_inputs


def get_batch_inputs(batch: List[str], vocab: Vocab, processor=None, max_len=None):
    """
    Encode the given batch to input to the model.

    Args:
        batch (list[str]): Batch of sequence,
            each sequence is represented by a string or list of tokens
        vocab (Vocab): Vocab of the batch
        processor (tokenizers.processors.PostProcessor): Optional, post-processor method
        max_len (int): Optional, the maximum length of each sequence

    Returns:
        (torch.LongTensor, torch.LongTensor): Tensor of batch and mask, [B, T]

    """
    # set post processor
    if processor:
        vocab.tokenizer.post_processor = processor
    else:
        vocab.tokenizer.post_processor = Vocab.void_processor
    # set truncation
    if max_len:
        vocab.tokenizer.enable_truncation(max_length=max_len)
    else:
        vocab.tokenizer.no_truncation()
    # encode batch
    inputs, padding_mask = vocab.encode_batch(batch, pad=True, max_length=max_len)
    # to tensor
    inputs = torch.tensor(inputs, dtype=torch.long)
    padding_mask = torch.tensor(padding_mask, dtype=torch.long)

    return inputs, padding_mask


def pad_batch(batch, pad_value=0):
    """
    Pad a list of sequence to a padded 2d tensor.

    Args:
        batch (list[list[int]]): List of sequence
        pad_value (int): Optional, fill value, default to 0.

    Returns:
        torch.Tensor: Padded tensor. [B, T].

    """
    batch = list(zip(*itertools.zip_longest(*batch, fillvalue=pad_value)))
    return torch.tensor([list(b) for b in batch]).long()
