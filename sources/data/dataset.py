import torch.utils.data
from torch.utils.data.dataset import Dataset

import os
import logging
import pickle
import random

import enums
from .data_utils import parse_for_summarization

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):

    def __init__(self, args, dataset_name, language, split):
        """
        Initialization definition.

        Args:
            args (argparse.Namespace): Arguments
            dataset_name (str): Name of the dataset
            language (str): Only for downstream fine-tuning
            split (str): Only for downstream fine-tuning, support ['train', 'valid', 'test', 'codebase']

        """
        super(CodeDataset, self).__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.language = language
        self.split = split
        self.paths = {}

        assert language in [enums.LANG_JAVA, enums.LANG_PYTHON]
        assert split in ['train', 'valid', 'test']
        self.dataset_dir = os.path.join(args.dataset_root, language, split)

        self.code_path = os.path.join(self.dataset_dir, 'split.token.code')
        self.nl_path = os.path.join(self.dataset_dir, 'raw.docstring')

        self.paths, self.codes, self.nls = parse_for_summarization(code_path=self.code_path, nl_path=self.nl_path)
        assert len(self.codes) == len(self.nls)
        self.size = len(self.codes)

    def __getitem__(self, index):
        return self.codes[index], self.nls[index]

    def __len__(self):
        return self.size

    def save(self):
        """Save to binary pickle file"""
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name}.pk')
        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')

    def subset(self, ratio):
        """
        Return a subset of self.

        Args:
            ratio (float): The ratio of size, must greater than 0 and less than/equal to 1

        Returns:
            Dataset: the subset

        """
        assert 0 < ratio <= 1, f'The subset ratio supposed to be 0 < ratio <= 1, but got ratio={ratio}'
        if ratio == 1:
            return self
        indices = random.sample(range(self.size), int(self.size * ratio))
        return torch.utils.data.Subset(self, indices)


def init_dataset(args, language=None, split=None, load_if_saved=True) -> CodeDataset:
    """
    Find dataset, if the dataset is saved, load and return, else initialize and return.

    Args:
        args (argparse.Namespace): Arguments
        language (str): Only for downstream fine-tuning
        split (str): Only for downstream fine-tuning, support ['train', 'valid', 'test', 'codebase(only for search)']
        load_if_saved (bool): Whether to load the saved instance if it exists, default to True

    Returns:
        CodeDataset: Loaded or initialized dataset

    """
    name = '.'.join([sub_name for sub_name in [language, split] if sub_name is not None])
    if load_if_saved:
        path = os.path.join(args.dataset_save_dir, f'{name}.pk')
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, CodeDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            print_paths(obj.paths)
            return obj
    dataset = CodeDataset(args=args,
                          dataset_name=name,
                          language=language,
                          split=split)
    dataset.save()
    return dataset


def print_paths(paths):
    """
    Print paths.

    Args:
        paths (dict): Dict mapping path group to path string or list of path strings.

    """
    logger.info('Dataset loaded from these files:')
    for key, value in paths.items():
        if isinstance(value, list):
            for v in value:
                logger.info(f'  {key}: {v}')
        else:
            logger.info(f'  {key}: {value}')


def save_all_datasets(args):
    # logger.info('*' * 100)
    # logger.info('Pre-training dataset')
    # _ = init_dataset(args=args,
    #                  mode=enums.TRAINING_MODE_PRE_TRAIN,
    #                  load_if_saved=False)
    # summarization
    for lang in [enums.LANG_JAVA, enums.LANG_GO, enums.LANG_PHP, enums.LANG_PYTHON, enums.LANG_RUBY,
                 enums.LANG_JAVASCRIPT]:
        for split in ['train', 'valid', 'test']:
            logger.info('*' * 100)
            logger.info(f'Summarization - {lang} - {split}')
            _ = init_dataset(args=args,
                             mode=enums.TRAINING_MODE_FINE_TUNE,
                             task=enums.TASK_SUMMARIZATION,
                             language=lang,
                             split=split,
                             load_if_saved=False)
    # # translation
    # for lang in ['java-c_sharp', 'c_sharp-java']:
    #     for split in ['train', 'valid', 'test']:
    #         logger.info('*' * 100)
    #         logger.info(f'Translation - {lang} - {split}')
    #         _ = init_dataset(args=args,
    #                          mode=enums.TRAINING_MODE_FINE_TUNE,
    #                          task=enums.TASK_TRANSLATION,
    #                          language=lang,
    #                          split=split,
    #                          load_if_saved=False)
    # # clone
    # from .data_utils import load_clone_mapping
    # code_mapping = load_clone_mapping(args.dataset_root)
    # for split in ['train', 'valid', 'test']:
    #     logger.info('*' * 100)
    #     logger.info(f'Clone - {split}')
    #     _ = init_dataset(args=args,
    #                      mode=enums.TRAINING_MODE_FINE_TUNE,
    #                      task=enums.TASK_CLONE_DETECTION,
    #                      split=split,
    #                      clone_mapping=code_mapping,
    #                      load_if_saved=False)
    # # # search
    # for lang in [enums.LANG_JAVA, enums.LANG_GO, enums.LANG_PHP, enums.LANG_PYTHON, enums.LANG_RUBY,
    #              enums.LANG_JAVASCRIPT]:
    #     for split in ['codebase', 'train', 'valid', 'test']:
    #         logger.info('*' * 100)
    #         logger.info(f'Search - {lang} - {split}')
    #         _ = init_dataset(args=args,
    #                          mode=enums.TRAINING_MODE_FINE_TUNE,
    #                          task=enums.TASK_SEARCH,
    #                          language=lang,
    #                          split=split,
    #                          load_if_saved=False)
    # # completion
    # for split in ['train', 'valid', 'test']:
    #     logger.info('*' * 100)
    #     logger.info(f'Completion - {split}')
    #     _ = init_dataset(args=args,
    #                      mode=enums.TRAINING_MODE_FINE_TUNE,
    #                      task=enums.TASK_COMPLETION,
    #                      split=split,
    #                      load_if_saved=False)
