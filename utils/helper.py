""" Contain all utilities """
import random
from typing import MutableSequence
from absl import flags

FLAGS = flags.FLAGS

def add_string(lhs: str, rhs: str, delimiter: str = " "):
    return delimiter.join([str(lhs), str(rhs)])

def merge_domain_name(domain: str, name: str) -> str:
    """ Merge domain name and other name, seperate by '-' """
    return f"{domain}-{name}"

def is_in_domain(domain: str, name: str) -> bool:
    return domain in name.split("-")[0]

def get_name(name: str) -> str:
    return name.split("-")[1]

def try_lowercase(text: str) -> str:
    """ Return string lowercase according to absl FLAGS """
    return text.lower() if FLAGS.lowercase is True else text

def try_shuffle(mutable: MutableSequence):
    """ Randomize mutable if flag random is on """
    if FLAGS.randomize_items:
        random.shuffle(mutable)