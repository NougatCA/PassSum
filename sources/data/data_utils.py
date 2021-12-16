
import logging
import random
import re

from asts.parse_paths import parse_paths


logger = logging.getLogger(__name__)


def load_lines(path):
    """
    Load lines from given path.

    Args:
        path (str): Dataset file path

    Returns:
        list: List of lines

    """
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f]
    return lines


def random_select(paths, count):
    """
    Randomly select given count of paths from all paths
    Args:
        paths (list):
        count (int):

    Returns:
        list
            - list of selected paths
    """
    indices = random.sample(range(len(paths)), k=count)
    indices.sort()
    return [paths[index] for index in indices]


def parse_for_summarization(code_path, nl_path, lang):
    """
    Load and parse dataset for code summarization.
    """
    assert lang in ['java', 'python']

    path_dict = {'code': code_path}
    logger.info(f'    Code file: {code_path}')
    codes = load_lines(code_path)

    path_dict['nl'] = nl_path
    logger.info(f'    Summarization file: {nl_path}')
    nls = load_lines(nl_path)

    codes, paths, nls = extract_paths(codes, nls, lang)

    assert len(codes) == len(paths) == len(nls)

    return path_dict, codes, paths, nls


def clean_code(code: str):
    """
    Clean code from whitespaces.
    Args:
        code:

    Returns:

    """
    return re.sub(r'\s+', ' ', code)


def extract_paths(codes, nls, lang):
    """
    Extract paths from code.
    """
    paths_list = []
    new_codes = []
    new_nls = []
    for code, nl in zip(codes, nls):
        try:
            paths = parse_paths(code, lang)
        except Exception:
            continue
        new_codes.append(clean_code(code))
        new_nls.append(nl)
        paths_list.append(paths)

    return new_codes, paths_list, new_nls
