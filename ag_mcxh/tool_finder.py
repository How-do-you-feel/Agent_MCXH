from typing import List
import numpy as np

from .apis.tool import list_tools

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity of a and b."""
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    res = dot_product / (norm_a * norm_b)
    return res

def _search_with_thefuzz(query, choices, topk=5):
    try:
        from thefuzz import process
        result = process.extract(query, choices=choices, limit=topk)
        return [res for res, _ in result]
    except ImportError:
        similarities = []
        for choice in choices:
            common_chars = len(set(query) & set(choice))
            similarities.append((choice, common_chars))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [choice for choice, _ in similarities[:topk]]

def search_tool(query: str, kind: str = 'thefuzz', topk=5) -> List[str]:
    """Search several proper tools according to the query.

    Args:
        query (str): User input.
        kind (str): Search method. Currently only supports "thefuzz".
        topk (int): Return the top-k results.

    Returns:
        list: List of tool names.
    """
    choice2names = dict()
    for name, description in list_tools(with_description=True):
        choice2names[description] = name

    choices = list(choice2names.keys())

    if kind == 'thefuzz':
        result = _search_with_thefuzz(query, choices, topk=topk)
    else:
        raise ValueError('The supported kind is "thefuzz"')

    names = []
    for description in result:
        names.append(choice2names[description])

    return names