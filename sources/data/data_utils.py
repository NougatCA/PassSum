
import logging
import os
import random


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


def load_java_paths(path_path, max_path_count=None):
    """
    Load java paths from dataset.
    Args:
        path_path (str): path to path file
        max_path_count (int):

    Returns:
        list:
            - list of list of paths in a sample, sorted by the leaf location, each path is represented by a tuple whose
              first element is a list of node name, and the second element is the leaf,
              e.g., (['MethodDeclaration', 'Body', ...], 'getName')
    """
    results = []
    with open(path_path, mode='r', encoding='utf-8') as f:
        for paths in f.readlines():
            tuples = []
            ps = paths.strip().split('#path#')
            if max_path_count and len(ps) > max_path_count:
                ps = random_select(ps, count=max_path_count)
            for p in ps:
                nodes_id = p.split('|')
                nodes = nodes_id[:-1]
                tuples.append((nodes, nodes_id[-1]))
            results.append(tuples)
    return results


def parse_python_paths(codes):
    pass


def parse_for_summarization(code_path, nl_path, path_path, lang):
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

    path_dict['path'] = path_path
    logger.info(f'    Path file: {path_path}')

    if lang == 'java':
        paths = load_java_paths(path_path)
    else:
        paths = parse_python_paths(codes)

    assert len(codes) == len(nls)

    return path_dict, codes, nls


def parse_paths(paths):
    pass
