""" Contain all utilities """
import random
from typing import MutableSequence
from absl import flags

import re

from natsort import humansorted

FLAGS = flags.FLAGS

# tag_pattern = r"^[*]$"


def add_string(lhs: str, rhs: str, delimiter: str = " ", first_empty=False) -> str:
    if lhs == str():
        return rhs
    if (
        first_empty and lhs.startswith("[") and lhs.endswith("]")
    ):  # re.match(tag_pattern, lhs):
        return " ".join([str(lhs), str(rhs)])
    return delimiter.join([str(lhs), str(rhs)])


def merge_domain_name(domain: str, name: str) -> str:
    """Merge domain name and other name, seperate by '-'"""
    return f"{domain}-{name}"


def is_in_domain(domain: str, name: str) -> bool:
    return domain in name.split("-")[0]


def get_name(name: str) -> str:
    name = name.split("-")
    if len(name) > 1:
        return name[1]
    else:
        return name


def try_lowercase(text: str) -> str:
    """Return string lowercase according to absl FLAGS"""
    return text.lower() if FLAGS.lowercase is True else text


def try_shuffle(mutable: MutableSequence):
    """Randomize mutable if flag random is on"""
    if FLAGS.randomize_items:
        random.shuffle(mutable)


def try_sort_id(mutable: MutableSequence, based_index=0, human_sorted=True):
    """Sort mutable based on item index if flag sort_id is on"""
    if FLAGS.sort_id:
        if human_sorted:
            # x2, x17, x20
            if based_index == -1:
                sorted_mutable = humansorted(mutable)
            else:
                sorted_mutable = humansorted(mutable, key=lambda x: x[based_index])
            for index, item in enumerate(sorted_mutable):
                mutable[index] = item
        else:
            # x17, x2, x20
            mutable.sort(key=lambda x: x[based_index])
