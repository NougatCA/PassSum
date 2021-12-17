
import tree_sitter
from tree_sitter import Language, Parser
import re

import enums


LANGUAGE = {enums.LANG_PYTHON: Language('build/my-languages.so', 'python'),
            enums.LANG_JAVA: Language('build/my-languages.so', 'java')}


parser = Parser()

SOURCE_PREFIX_POSTFIX = {
    enums.LANG_JAVA: ['class A{ ', ' }']
}

PATTERNS_METHOD_ROOT = {
    enums.LANG_JAVA: """
    (program
        (class_declaration
            body: (class_body
                (method_declaration) @method_root)
        )
    )
    """
}

ELIMINATE_TERMINAL = ['(', ')', '{', '}', ';', ',', '[', ']', '<', '>', '\"', '\'', ':', '@']


def camel_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def split_identifier(identifier):
    """
    Split identifier into a list of subtokens.
    Tokens except characters and digits will be eliminated.

    Args:
        identifier (str): given identifier

    Returns:
        list[str]: list of subtokens
    """
    words = []

    word = re.sub(r'[^a-zA-Z0-9]', ' ', identifier)
    word = re.sub(r'(\d+)', r' \1 ', word)
    split_words = word.strip().split()
    for split_word in split_words:
        camel_words = camel_split(split_word)
        for camel_word in camel_words:
            words.append(camel_word.lower())

    return words


def get_node_name(source, node, lang):
    """
    Get node name, for php is shifted by prefix.

    Args:
        source (str): Source code string
        node (tree_sitter.Node): Node instance
        lang (str): Source code language

    Returns:
        str: Name of node

    """
    if node.is_named:
        if lang in SOURCE_PREFIX_POSTFIX:
            return source[node.start_byte - len(SOURCE_PREFIX_POSTFIX[lang][0]):
                          node.end_byte - len(SOURCE_PREFIX_POSTFIX[lang][0])]
        else:
            return source[node.start_byte: node.end_byte]
    return ''


def parse_ast(source, lang):
    """
    Parse the given code into corresponding ast.
    Args:
        source (str): code in string
        lang (str): Set the language

    Returns:
        tree_sitter.Node: Method/Function root node

    """
    parser.set_language(LANGUAGE[lang])
    if lang in SOURCE_PREFIX_POSTFIX:
        source = SOURCE_PREFIX_POSTFIX[lang][0] + source + SOURCE_PREFIX_POSTFIX[lang][1]
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    root = tree.root_node
    if lang in PATTERNS_METHOD_ROOT:
        query = LANGUAGE[lang].query(PATTERNS_METHOD_ROOT[lang])
        captures = query.captures(root)
        root = captures[0][0]
    return root


def get_all_paths(root: tree_sitter.Node):
    """
    Get all paths from root to each terminal node.

    Args:
        root (tree_sitter.Node): Method/Function root node

    Returns:
        list[list[tree_sitter.Node]]: List of paths from root to each terminal node

    """
    node_stack = []
    path_stack = []
    result = []

    node_stack.append(root)
    init_path = [root]
    path_stack.append(init_path)

    while node_stack:
        cur_node = node_stack.pop()
        cur_path = path_stack.pop()

        if cur_node.child_count <= 0:
            if cur_node.type not in ELIMINATE_TERMINAL:
                result.append(cur_path)
        else:
            num_children = len(cur_node.children)
            for i in range(num_children - 1, -1, -1):
                child = cur_node.children[i]
                node_stack.append(child)
                new_path = cur_path[:]
                new_path.append(child)
                path_stack.append(new_path)

    return result


def parse_paths(source, lang):
    """
    Parse the given paths into corresponding (paths, id) tuple.
    Args:
        source (str): code in string
        lang (str): Set the language

    Returns:
        tuple_paths (List[Tuple[List[str], List[str]]]): List of (paths, id) tuple

    """
    root = parse_ast(source, lang)
    paths = get_all_paths(root)

    tuple_paths = []
    for path in paths:
        nodes = [node.type for node in path]
        ids = split_identifier(get_node_name(source, path[-1], lang))
        if len(ids) == 0:
            ids = [nodes[-1]]
            nodes.pop()
        tuple_paths.append((nodes, ids))
    return tuple_paths


# def lang_sample(lang):
#     import random
#     with open(f'../../../../dataset/{lang}/valid/source.code') as f:
#         line = f.readlines()[random.randint(0, 1000)]
#     return line
#
#
# lang = 'java'
#
# source = lang_sample(lang)
# print('-' * 100)
# print('source:')
# print(source)
# print('-' * 100)
#
# root = parse_ast(source, lang=lang)
# paths = get_all_paths(root)
# tuple_paths = parse_paths(paths, source, lang)
# for path in tuple_paths:
#     print(path)
