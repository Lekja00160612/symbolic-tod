r"""
    Schema is missing possible user and system actions. 
    This module serve the purpose by iterating through every dialogues.
    Add 2 new attribute to each service schema:
    -   "possible_user_actions": [string]
    -   "possible_system_actions": [string]   
"""
from collections import defaultdict
import re
import os
import json
import pathlib
from typing import Dict, Set, Tuple
from absl import app
from absl import logging
from absl import flags
from utils import ActionTemplate, NON_DOMAIN_USER_ACTIONS, NON_DOMAIN_SYSTEM_ACTIONS

logging.set_verbosity("info")

FLAGS = flags.FLAGS
flags.DEFINE_string("schema_file", None, "Schema file path.")
flags.DEFINE_string("out_schema_file", None, "Output schema file path.")
flags.DEFINE_string("log_folder", None, "Log folder path.")
flags.DEFINE_string(
    "sgd_folder",
    None,
    "The SGD folder path. (path to one of 'train', 'dev', 'test')",
)
flags.DEFINE_boolean(
    "insert_space", False, "turn '_' to ' ' and add space between capitalized part"
)

flags.DEFINE_boolean(
    "include_query_action", True, "whether to consider calldb as an action"
)

SchemaAction = Set[Tuple[str, str]]
AllSchemasAction = Dict[str, SchemaAction]


def resolve_user_action(act, slot, value, possible_user_actions):
    """
    Resolve single user action based on ActionTemplate
    """
    act_desc = ""
    match act:
        case "inform":
            act_desc = ActionTemplate.USER_INFORM.format(slot_name=slot)
        case "request":
            act_desc = ActionTemplate.USER_REQUEST.format(slot_name=slot)
        case "select":
            # TODO: replace with some other slot_name
            act_desc = ActionTemplate.USER_SELECT.format(
                slot_name=""  # if slot == "" else slot
            ).strip()
            # if slot != "":
            #     logging.warning(f"Resolver did not handle select with {slot=}")
        case "inform_intent":
            act_desc = ActionTemplate.USER_INTENT.format(intent_name=value)
            slot = value
        case "negate" | "negate_intent":
            act = "negate"
            act_desc = ActionTemplate.USER_NEGATE
        case "affirm" | "affirm_intent":
            act = "affirm"
            act_desc = ActionTemplate.USER_AFFIRM
        case "thank_you":
            act_desc = ActionTemplate.USER_THANK_YOU
        case "goodbye":
            act_desc = ActionTemplate.USER_GOODBYE
        case "request_alts":
            act_desc = ActionTemplate.USER_REQUEST_ALTS
        case _:
            logging.error(f"Resolver did not handle user {act=} with {slot=}")
    possible_user_actions.add(
        (f"{act}_{slot}" if (slot not in ["", "intent"]) else f"{act}", act_desc)
    )
    return possible_user_actions


def resolve_system_action(act, slot, value, possible_system_actions):
    """
    Resolve single system action based on ActionTemplate
    """
    act_desc = ""
    match act:
        case "confirm":
            act_desc = ActionTemplate.SYSTEM_CONFIRM.format(slot_name=slot)
        case "inform_count":
            act_desc = ActionTemplate.SYSTEM_INFORM_COUNT
        case "offer":
            act_desc = ActionTemplate.SYSTEM_OFFER.format(slot_or_intent_name=slot)
        case "offer_intent":
            act_desc = ActionTemplate.SYSTEM_OFFER.format(slot_or_intent_name=value)
            slot = value
        case "inform":
            act_desc = ActionTemplate.SYSTEM_INFORM.format(slot_name=slot)
        case "request":
            act_desc = ActionTemplate.SYSTEM_REQUEST.format(slot_name=slot)
        case "goodbye":
            act_desc = ActionTemplate.SYSTEM_GOODBYE
        case "notify_success":
            act_desc = ActionTemplate.SYSTEM_NOTIFY_SUCCESS
        case "notify_failure":
            act_desc = ActionTemplate.SYSTEM_NOTIFY_FAILURE
        case "req_more":
            act_desc = ActionTemplate.SYSTEM_REQ_MORE
        case _:
            logging.error(f"Resolver did not handle system {act=} with {slot=}")
    possible_system_actions.add(
        (
            f"{act}_{slot}" 
            if (slot not in ["", "intent", "count"]) or (act == "request" and slot == "intent") 
            else f"{act}",
            act_desc,
        )
    )
    return possible_system_actions

def handle_exception_cases(domain_name, schema, possible_user_actions, possible_system_actions):
    if "movies_1" in domain_name:
        # In dataset, user never affirm to book tickets, so navigating in the dataset only would result in missing actions
        for intent in schema["intents"]:
            if "ticket" in intent["name"].lower() or "seat" in intent["name"].lower(): 
                for slot in intent["required_slots"]:
                    possible_user_actions[domain_name].add(
                        (
                            f"inform_{slot}",
                            ActionTemplate.USER_INFORM.format(
                                slot_name=slot
                            ),
                        )
                    )
                    possible_system_actions[domain_name].update(
                        [
                            (
                                f"request_{slot}",
                                ActionTemplate.SYSTEM_REQUEST.format(
                                    slot_name=slot
                                ),
                            ),
                            (
                                f"confirm_{slot}",
                                ActionTemplate.SYSTEM_CONFIRM.format(
                                    slot_name=slot
                                ),
                            ),
                        ]
                    )
    elif "homes_2" in domain_name:
        if "intent" in list(slot["name"] for slot in schema["slots"]):
            logging.info(domain_name)
            possible_system_actions[domain_name].add((
                "request_intent",
                ActionTemplate.SYSTEM_REQUEST.format(slot_name="intent")
            ))
            possible_user_actions[domain_name].add((
                "inform_intent",
                ActionTemplate.USER_INFORM.format(slot_name="intent")
            ))

    return possible_user_actions, possible_system_actions

def generate_schema_with_action() -> AllSchemasAction:
    """
    Iterate through every schema, include possible actions
    """
    # Init default dict with default value as a set
    possible_user_actions = defaultdict(lambda: set())
    possible_system_actions = defaultdict(lambda: set())
    dialogues_dir = pathlib.Path(FLAGS.sgd_folder)
    # Iterate through every files contain dialogues
    for dialogues_file in sorted(dialogues_dir.rglob("dialogues_*.json")):
        logging.info(f"processing {dialogues_file}")
        with open(dialogues_file, "r", encoding="utf-8") as file:
            dialogues = json.load(file)
            # Iterate through each dialogue
            for dialogue in dialogues:
                # A dialogue has multiple turns with speaker attribute
                for turn in dialogue["turns"]:
                    speaker = turn["speaker"].lower()
                    # A turn has multiple frames, each represent an domain
                    # With single domain dialogue, each turn only has 1 frame
                    for frame in turn["frames"]:
                        domain_name = frame["service"].lower()
                        # Each frame may has multiple action, must convert all to symbol
                        for action in frame["actions"]:
                            act = action["act"].lower()
                            # Check if insert_space flag is not on, keep the original value
                            slot = (
                                action["slot"].lower()
                                if FLAGS.insert_space is False
                                else action["slot"].lower.replace("_", " ")
                            )
                            # only use when the action is related to intent
                            # some turns might not have value, for example, 
                            # "How many beds and in what area do you want the property in? Would you like to rent or buy?"
                            
                            if slot == "intent" and action["act"].lower() in ["request", "inform"]: # len(action["values"]) == 0:
                                assert "homes_2" in domain_name
                                logging.info(f"{str(dialogue['dialogue_id'])} got intent as slot")
                                continue
                            
                            if slot == "intent" and len(action["values"]) > 0:
                                value = action["values"][0].lower()
                            else:
                                value = None

                            if value is not None:
                                value = (
                                    value
                                    if FLAGS.insert_space is False
                                    else re.sub(
                                        r"(\w)([A-Z])", r"\1 \2", action["values"][0]
                                    ).lower()
                                )
                            if speaker == "user":
                                possible_user_actions[
                                    domain_name
                                ] = resolve_user_action(
                                    act, slot, value, possible_user_actions[domain_name]
                                )
                            elif speaker == "system":
                                possible_system_actions[
                                    domain_name
                                ] = resolve_system_action(
                                    act,
                                    slot,
                                    value,
                                    possible_system_actions[domain_name],
                                )

    schemas = []
    with open(FLAGS.schema_file, "r", encoding="utf-8") as sm_file:
        for schema in json.load(sm_file):
            domain_name = schema["service_name"].lower()
            # Handle query action
            for intent in schema["intents"]:
                intent = intent["name"].lower()
                possible_system_actions[domain_name].add(
                    (
                        ActionTemplate.KEY_SYSTEM_QUERY.format(intent_name=intent),
                        ActionTemplate.SYSTEM_QUERY.format(intent_name=intent),
                    )
                )
            # Handle exceptions, these cases were explored during data generation
            possible_user_actions, possible_system_actions = handle_exception_cases(
                                                                        domain_name, 
                                                                        schema, 
                                                                        possible_user_actions, 
                                                                        possible_system_actions
                                                                    )
                
            possible_system_actions[domain_name].update(
                [
                    (
                        "_".join(action.split("_")[1:]),
                        getattr(ActionTemplate, action.upper()),
                    )
                    for action in NON_DOMAIN_SYSTEM_ACTIONS
                ]
            )
            possible_user_actions[domain_name].update(
                [
                    (
                        "_".join(action.split("_")[1:]),
                        getattr(ActionTemplate, action.upper()),
                    )
                    for action in NON_DOMAIN_USER_ACTIONS
                ]
            )

            schema["possible_user_actions"] = {
                pair[0]: pair[1] for pair in possible_user_actions[domain_name]
            }
            schema["possible_system_actions"] = {
                pair[0]: pair[1] for pair in possible_system_actions[domain_name]
            }
            schemas.append(schema)
    directory = os.path.split(FLAGS.out_schema_file)[0]
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(FLAGS.out_schema_file, "w", encoding="utf-8") as out_sm_file:
        json.dump(schemas, out_sm_file, indent=2)

def set_log_dir():
    if not os.path.exists(FLAGS.log_folder):
        os.makedirs(FLAGS.log_folder)
    logging.get_absl_handler().use_absl_log_file("absl_logging", FLAGS.log_folder)


def main(_):
    """
    Main execution
    """
    set_log_dir()
    generate_schema_with_action()


if __name__ == "__main__":
    flags.mark_flag_as_required("sgd_folder")
    flags.mark_flag_as_required("schema_file")
    flags.mark_flag_as_required("out_schema_file")
    app.run(main)
