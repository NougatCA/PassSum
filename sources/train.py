
from transformers import BartForConditionalGeneration
from tokenizers import Tokenizer

import logging
from typing import Union, Tuple

from data.vocab import Vocab
from downstream_tasks.summarization import run_summarization

logger = logging.getLogger(__name__)


def train(
        args,
        trained_model: Union[BartForConditionalGeneration, str] = None,
        trained_vocab: Union[Vocab, str] = None):
    """
    Fine-tuning from given pre-trained model and vocabs, or training from scratch.

    Args:
        args (argparse.Namespace): Arguments
        trained_model (Union[BartForConditionalGeneration, str]): Optional,
            instance or directory of ``BartForClassificationAndGeneration``, must given when ``only_test`` is True
        trained_vocab (Union[Tokenizer, str]): Optional, Tuple of instances or directory of three
            vocabularies, must given when ``only_test`` is True

    """
    if trained_model is None and args.trained_model is not None:
        trained_model = args.trained_model
    assert not args.only_test or isinstance(trained_model, str) or \
           isinstance(trained_model, BartForConditionalGeneration), \
           f'The model type is not supported, expect Bart model or string of model dir, got {type(trained_model)}'

    if trained_vocab is None and args.trained_vocab is not None:
        trained_vocab = args.trained_vocab
    assert not args.only_test or isinstance(trained_vocab, str) or isinstance(trained_vocab, Vocab), \
        f'The vocab type is not supported, expect Vocab or string of path, got {type(trained_vocab)}'

    logger.info('*' * 100)
    if trained_model:
        logger.info('Fine-tuning from pre-trained model and vocab')
        if isinstance(trained_model, str):
            logger.info(f'Model dir: {trained_model}')
        if isinstance(trained_vocab, str):
            logger.info(f'Vocab dir: {trained_vocab}')
    else:
        logger.info('Training from scratch')

    run_summarization(args=args,
                      trained_model=trained_model,
                      trained_vocab=trained_vocab,
                      only_test=args.only_test)
