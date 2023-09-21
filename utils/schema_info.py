""" define what can be in the schema """
from dataclasses import field, dataclass
from typing import Dict, Tuple
from collections import defaultdict
import json

from absl import flags, logging

from utils import ActionTemplate
from utils.helper import merge_domain_name, try_lowercase

FLAGS = flags.FLAGS


@dataclass
class SchemaInfo:
    """Schema information"""

    # Action related
    possible_user_actions: Dict[str, str] = field(default_factory=dict)
    possible_system_actions: Dict[str, str] = field(default_factory=dict)

    # Param related
    possible_values: Dict[str, str] = field(default_factory=dict)
    is_categorical: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, str] = field(default_factory=dict)

    # Intent related
    is_transactional: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)
    system_offer_params: Dict[str, str] = field(default_factory=dict)


def retrieve_dependencies_slots(schema, rule_based=False):
    user_inform_slots = defaultdict(lambda: [])
    system_offer_slots = defaultdict(lambda: [])
    system_confirm_slots = defaultdict(lambda: [])
    for intent in schema["intents"]:
        other_intents = [
            other_intent
            for other_intent in schema["intents"]
            if other_intent["name"] != intent["name"]
        ]

        intent_name = intent["name"].lower()

        for required_slot in intent["required_slots"]:
            # --------------- Rule-based which fail ------------------ #
            # There are slots that can either be user_inform or system_offer, no rule for these
            if rule_based:
                # If the intent is not transactional then all slots are from user_inform
                if intent["is_transactional"] is False:
                    user_inform_slots[intent_name].append(required_slot)
                    continue

                # From here, we process transactional intent only
                # or required_slot cannot be found in any other_intent result_slots
                if not any(
                    required_slot in other_intent["result_slots"]
                    for other_intent in other_intents
                ):
                    user_inform_slots[intent_name].append(required_slot)
                system_confirm_slots[intent_name].append(required_slot)
                for other_intent in other_intents:
                    if required_slot in other_intent["result_slots"]:
                        if (
                            required_slot
                            in other_intent["required_slots"]
                            # or required_slot in other_intent["optional_slots"]
                        ):
                            user_inform_slots[intent_name].append(required_slot)
                        # Not in both, it means must come from system_offer
                        else:
                            system_offer_slots[intent_name].append(required_slot)
                # Validate slots availability
                for intent in schema["intents"]:
                    intent_name = intent["name"].lower()
                    try:
                        [
                            f"system_confirm_{required_slot}"
                            in schema["possible_system_actions"]
                            for required_slot in system_confirm_slots[intent_name]
                        ]
                    except Exception:
                        logging.fatal(
                            f"{system_confirm_slots} not exist in {schema['possible_system_actions']}"
                        )
                    try:
                        [
                            f"system_offer_{required_slot}"
                            in schema["possible_system_actions"]
                            for required_slot in system_offer_slots[intent_name]
                        ]
                    except Exception:
                        logging.fatal(
                            f"{system_offer_slots} not exist in {schema['possible_system_actions']}"
                        )
                    try:
                        [
                            f"user_inform_{required_slot}"
                            in schema["possible_user_actions"]
                            for required_slot in user_inform_slots[intent_name]
                        ]
                    except Exception:
                        logging.fatal(
                            f"{user_inform_slots} not exist in {schema['possible_user_actions']}"
                        )
            # Try heuristic methods based on possible actions
            else:
                if intent["is_transactional"] is False:
                    user_inform_slots[intent_name].append(required_slot)
                    if f"inform_{required_slot}" not in schema["possible_user_actions"]:
                        logging.fatal(
                            f"inform_{required_slot} not in {schema['possible_user_actions']}"
                        )
                else:
                    system_confirm_slots[intent_name].append(required_slot)
                    if (
                        f"confirm_{required_slot}"
                        not in schema["possible_system_actions"]
                    ):
                        logging.fatal(
                            f"confirm_{required_slot} not in {schema['possible_system_actions']}"
                        )

                    if f"inform_{required_slot}" in schema["possible_user_actions"]:
                        user_inform_slots[intent_name].append(required_slot)

                    if f"offer_{required_slot}" in schema["possible_system_actions"]:
                        system_offer_slots[intent_name].append(required_slot)

                    if (
                        required_slot in system_offer_slots[intent_name]
                        and required_slot in user_inform_slots[intent_name]
                    ):
                        logging.info(
                            f"{required_slot} can come from both user_inform and system_offer"
                        )
    logging.info(
        f"{json.dumps(schema['intents'],indent=4)}\n\tuser_inform_slots: {json.dumps(user_inform_slots,indent=4)}\n\tsystem_offer_slots: {json.dumps(system_offer_slots,indent=4)}\n\tsystem_confirm_slots: {json.dumps(system_confirm_slots,indent=4)}"
    )


def load_schema() -> SchemaInfo:
    """Loads schema items and descriptions.

    Returns:
      A tuple, including an ordered dictionary whose keys are slot names and
      values are placeholder values, and a dictionary whose keys are slot names
      and values are descriptions.
    """
    # We need to preserve state orders since we hope the model learns to generate
    # states in consistent order.
    # slots = collections.OrderedDict()
    item_desc = SchemaInfo()
    with open(FLAGS.schema_file, "r", encoding="utf-8") as sm_file:
        for schema in json.load(sm_file):
            domain = (
                schema["service_name"].lower()
                if FLAGS.lowercase
                else schema["service_name"]
            )
            # slots.update(
            #     {
            #         merge_domain_name(domain, slot["name"]): ""
            #         for slot in schema["slots"]
            #     }
            # )
            item_desc.params.update(
                {
                    merge_domain_name(domain, slot["name"]): slot["description"]
                    for slot in schema["slots"]
                }
            )

            for slot in schema["slots"]:
                name = merge_domain_name(domain, slot["name"])
                is_categorical = slot["is_categorical"]
                poss_vals = slot["possible_values"]

                # If this is a categorical slot but the possible value are all numeric,
                # consider this as a noncat slot.
                if is_categorical and all(v.isdigit() for v in poss_vals):
                    poss_vals = []
                    is_categorical = False

                item_desc.is_categorical[name] = is_categorical
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

            # get intent related information
            if len(schema["intents"]) > 2:
                logging.info(f"{domain} has more than 2 intents!")

            for intent in schema["intents"]:
                intent_name = merge_domain_name(
                    domain,
                    f"system_{ActionTemplate.KEY_SYSTEM_QUERY.format(intent_name=try_lowercase(intent['name']))}",
                )

                item_desc.dependencies[intent_name] = intent["required_slots"]

                item_desc.is_transactional[intent_name] = intent["is_transactional"]

            # get slot strict level
            # retrieve_dependencies_slots(schema=schema)

    return item_desc


if __name__ == "__main__":
    print(load_schema()[1].dependencies)
