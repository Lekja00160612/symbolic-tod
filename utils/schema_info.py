""" define what can be in the schema """
from dataclasses import field, dataclass
from typing import Dict, Tuple
import collections
import json

from absl import flags

from utils import ActionTemplate
from utils.helper import merge_domain_name

FLAGS = flags.FLAGS

@dataclass
class SchemaInfo:
    """Schema information"""

    possible_user_actions: Dict[str, str] = field(default_factory=dict)
    possible_system_actions: Dict[str, str] = field(default_factory=dict)
    possible_values: Dict[str, str] = field(default_factory=dict)
    is_categorical: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)

def load_schema() -> Tuple[collections.OrderedDict, SchemaInfo]:
    """Loads schema items and descriptions.

    Returns:
      A tuple, including an ordered dictionary whose keys are slot names and
      values are placeholder values, and a dictionary whose keys are slot names
      and values are descriptions.
    """
    # We need to preserve state orders since we hope the model learns to generate
    # states in consistent order.
    # TODO(yuancao): We might need to do this for intents/actions as well (in case
    # multiple intents/actions turn out to be a problem).
    # TODO(jeffreyzhao): Clean up how we store schema information by using a
    # dataclass.
    slots = collections.OrderedDict()
    item_desc = SchemaInfo()
    with open(FLAGS.schema_file, "r", encoding="utf-8") as sm_file:
        for schema in json.load(sm_file):
            domain = (
                schema["service_name"].lower()
                if FLAGS.lowercase
                else schema["service_name"]
            )
            slots.update(
                {
                    merge_domain_name(domain, slot["name"]): ""
                    for slot in schema["slots"]
                }
            )
            item_desc.params.update(
                {
                    merge_domain_name(domain, slot["name"]): slot["description"]
                    for slot in schema["slots"]
                }
            )

            for slot in schema["slots"]:
                name = merge_domain_name(domain, slot["name"])
                is_cat = slot["is_categorical"]
                poss_vals = slot["possible_values"]

                # If this is a categorical slot but the possible value are all numeric,
                # consider this as a noncat slot.
                if is_cat and all(v.isdigit() for v in poss_vals):
                    poss_vals = []
                    is_cat = False

                item_desc.is_categorical[name] = is_cat
                item_desc.possible_values[name] = poss_vals

            item_desc.possible_system_actions.update(
                {
                    merge_domain_name(domain, f"system_{action_name}"): action_desc
                    for action_name, action_desc in schema[
                        "possible_system_actions"
                    ].items()
                }
            )

            item_desc.possible_user_actions.update(
                {
                    merge_domain_name(domain, f"user_{action_name}"): action_desc
                    for action_name, action_desc in schema[
                        "possible_user_actions"
                    ].items()
                }
            )

            item_desc.dependencies.update(
                {
                    merge_domain_name(
                        domain,
                        f"system_{ActionTemplate.QUERY_NAME.format(intent_name=intent['name'].lower() if FLAGS.lowercase else intent['name'])}",
                    ): intent["required_slots"]
                    for intent in schema["intents"]
                }
            )
    return slots, item_desc