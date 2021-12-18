
import torch

from typing import List
import itertools
import numpy as np

from data.vocab import Vocab


def collate_fn(batch, args, code_vocab, node_vocab):
    """
    Data collator function.
    """
    model_inputs = {}

    code_raw, path_raw, nl_raw = map(list, zip(*batch))

    model_inputs['code_input_ids'], code_attention_mask = get_batch_inputs(
        batch=code_raw,
        vocab=code_vocab,
        processor=Vocab.sep_processor,
        max_len=args.max_code_len
    )

    path_raw = random_sample_path(path_raw, args.max_path_len)

    model_inputs['path_seq_lens'] = get_seq_lens(path_raw)
    path_attention_mask = generate_attention_mask(model_inputs['path_seq_lens'])

    model_inputs['attention_mask'] = torch.cat([code_attention_mask, path_attention_mask], dim=-1)

    id_inputs = []
    node_inputs = []
    for tuples in path_raw:
        for (nodes, ids) in tuples:
            node_inputs.append(nodes)
            id_inputs.append(ids)
    id_inputs = indices_from_batch(batch=id_inputs,
                                   vocab=code_vocab,
                                   add_eos=True,
                                   max_length=args.max_id_len)
    model_inputs['id_seq_lens'] = get_seq_lens(id_inputs)
    model_inputs['id_inputs'] = pad_batch(id_inputs)  # [B, T]

    node_inputs = indices_from_batch(batch=node_inputs,
                                     vocab=node_vocab,
                                     add_eos=True,
                                     max_length=args.max_node_len)
    model_inputs['node_seq_lens'] = get_seq_lens(node_inputs)
    model_inputs['node_inputs'] = pad_batch(node_inputs)  # [B, T]

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


def get_seq_lens(batch: list):
    """
    Returns the sequence lengths of given batch.
    Args:
        batch (list): given batch

    Returns:
        list:
            - sequence length of each sequence
    """
    return [len(seq) for seq in batch]


def indices_from_batch(batch, vocab: Vocab, add_sos=False, add_eos=False, max_length=None):
    """
    Translate the word in batch to corresponding index by given vocab, and add EOS token to the end of each sentence
    Args:
        batch (list): batch to be translated, [B, ~T]
        vocab (vocab.Vocab): Vocab
        add_sos (bool): True when need to add SOS token to the start of each sentence
        add_eos (bool): True when need to add EOS token to the end of each sentence
        max_length (int):

    Returns:
        list:
            - translated batch, [B, ~T]
    """
    if add_sos and add_eos:
        reserve_n_token = 2
    elif not add_eos and add_sos:
        reserve_n_token = 0
    else:
        reserve_n_token = 1

    indices = []
    for sentence in batch:
        if max_length and len(sentence) > max_length - reserve_n_token:
            sentence = sentence[:max_length - reserve_n_token]
        indices_sentence = []
        if add_sos:
            indices_sentence.append(vocab.get_sos_index())
        for word in sentence:
            indices_sentence.append(vocab.get_index(word))
        if add_eos:
            indices_sentence.append(vocab.get_eos_index())
        indices.append(indices_sentence)
    return indices


def generate_attention_mask(seq_lens):
    """
    Generate attention mask from sequence lengths.
    Args:
        seq_lens (list): sequence lengths

    Returns:
        torch.Tensor: attention mask, [B, T]
    """
    attention_mask = torch.zeros(len(seq_lens), max(seq_lens)).long()
    for i, seq_len in enumerate(seq_lens):
        attention_mask[i, :seq_len] = 1
    return attention_mask


def random_sample_path(path_raw, max_path_len):
    """
    Randomly sample path from path_raw.
    Args:
        path_raw (list): path raw
        max_path_len (int): max path length

    Returns:
        list: sampled path
    """
    paths = []
    for path in path_raw:
        if len(path) > max_path_len:
            indices = np.random.choice(len(path), max_path_len, replace=False)
            indices = sorted(indices)
            paths.append([path[i] for i in indices])
        else:
            paths.append(path)
    return paths
