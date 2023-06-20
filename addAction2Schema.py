r"""
    Schema is missing possible user and system actions. 
    This module serve the purpose by iterating through every dialogues.
    Add 2 new attribute to each service schema:
    -   "possible_user_actions": [string]
    -   "possible_system_actions": [string]   
"""
from collections import defaultdict
import re
import json
import pathlib
from enum import StrEnum
from typing import Dict, Set, Tuple
from absl import app
from absl import logging
from absl import flags
from actionTemplate import ActionTemplate

logging.set_verbosity("info")

FLAGS = flags.FLAGS
flags.DEFINE_string("schema_file", None, "Schema file path.")
flags.DEFINE_string("out_schema_file", None, "Output schema file path.")
flags.DEFINE_string(
    "sgd_folder",
    None,
    "The SGD folder path. (path to one of 'train', 'dev', 'test')",
)
flags.DEFINE_boolean("insert_space", False, "turn '_' to ' ' and add space between capitalized part")

flags.DEFINE_boolean("include_query_action", True, "whether to consider calldb as an action")

SchemaAction = Set[Tuple[str,str]]
AllSchemasAction = Dict[str, SchemaAction]

def resolve_user_action(act, slot, value, possible_user_actions):
    act_desc = ""
    match act:
        case "inform":
            act_desc = ActionTemplate.USER_INFORM.format(slot_name=slot)
        case "request":
            act_desc = ActionTemplate.USER_REQUEST.format(slot_name=slot)
        case "select":
            # TODO: replace with some other slot_name
            act_desc = ActionTemplate.USER_SELECT.format(slot_name="" if slot == "" else slot).strip()
            # if slot != "":
            #     logging.warning(f"Resolver did not handle select with {slot=}")
        case "inform_intent":
            act_desc = ActionTemplate.USER_INTENT.format(intent_name=value)
            slot = value
        case "negate" | "negate_intent":
            act_desc = ActionTemplate.USER_NEGATE
        case "affirm" | "affirm_intent":
            act_desc = ActionTemplate.USER_AFFIRM
        case "thank_you":
            act_desc = ActionTemplate.USER_THANKYOU
        case "goodbye":
            act_desc = ActionTemplate.USER_GOODBYE
        case "request_alts":
            act_desc = ActionTemplate.USER_REQUEST_ALTS
        case _:
            logging.error(f"Resolver did not handle user {act=} with {slot=}")
    possible_user_actions.add(
        (
            f"{act}_{slot}" if (slot not in ["","intent"]) else f"{act}",
            act_desc
        )
    )
    return possible_user_actions

def resolve_system_action(act, slot, value, possible_system_actions):
    act_desc = ""
    match act:
        case "confirm":
            act_desc = ActionTemplate.SYSTEM_CONFIRM.format(slot_name=slot)
        case "inform_count":
            act_desc = ActionTemplate.SYSTEM_INFORMCOUNT
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
            act_desc = ActionTemplate.SYSTEM_REQMORE
        case _:
            logging.error(f"Resolver did not handle system {act=} with {slot=}")
    possible_system_actions.add(
        (
            f"{act}_{slot}" if (slot not in ["","intent"]) else f"{act}",
            act_desc
        )
    )
    return possible_system_actions

def generate_schema_with_action() -> AllSchemasAction:
    # Init default dict with default value as a set
    possible_user_actions = defaultdict(lambda: set())
    possible_system_actions = defaultdict(lambda: set())
    dialogues_dir = pathlib.Path(FLAGS.sgd_folder)
    # Iterate through every files contain dialogues
    for dialogues_file in dialogues_dir.rglob("dialogues_*.json"):
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
                        domain_name = frame["service"]
                        # Each frame may has multiple action, must convert all to symbol
                        for action in frame["actions"]:
                            act = action["act"].lower()
                            # Check if insert_space flag is not on, keep the original value
                            slot = action["slot"].lower() if FLAGS.insert_space is False else action["slot"].lower.replace("_"," ")
                            # only use when the action is related to intent
                            value = action["values"][0].lower() if slot == "intent" else None
                            if value is not None:
                                value = action["values"][0].lower() if FLAGS.insert_space is False else re.sub(r"(\w)([A-Z])", r"\1 \2", action["values"][0]).lower() if slot == "intent" else None
                            if speaker == "user":
                                # if dialogue["dialogue_id"] == "11_00002":
                                #     logging.error(f"{act}_{slot}")
                                #     logging.error(f"{possible_user_actions[domain_name]}")
                                possible_user_actions[domain_name] = resolve_user_action(act,slot,value,possible_user_actions[domain_name])                            
                            elif speaker == "system":
                                possible_system_actions[domain_name] = resolve_system_action(act,slot,value,possible_system_actions[domain_name])

    schemas = []
    with open(FLAGS.schema_file, "r", encoding="utf-8") as sm_file:
        for schema in json.load(sm_file):
            domain_name = schema["service_name"]
            # Handle query action
            for intent in schema["intents"]:
                intent = intent["name"].lower()
                possible_system_actions[domain_name].add((ActionTemplate.QUERY_NAME.format(intent_name=intent),(ActionTemplate.SYSTEM_QUERY.format(intent_name=intent))))
            # Handle exceptions
            if "Movies_1" in domain_name:
                # In dataset, user never affirm to book tickets, so navigating in the dataset only would result in missing actions
                possible_user_actions[domain_name].add(("inform_number_of_tickets","user informing number_of_tickets"))
                possible_system_actions[domain_name].update([("request_number_of_tickets", ActionTemplate.SYSTEM_REQUEST.format(slot_name="number_of_tickets")),
                                                             ("confirm_number_of_tickets", ActionTemplate.SYSTEM_CONFIRM.format(slot_name="number_of_tickets"))])
            
            schema["possible_user_actions"] = {pair[0]: pair[1] for pair in possible_user_actions[domain_name]}
            schema["possible_system_actions"] = {pair[0]: pair[1] for pair in possible_system_actions[domain_name]}
            schemas.append(schema)
    with open(FLAGS.out_schema_file, "w", encoding="utf-8") as out_sm_file:
        json.dump(schemas,out_sm_file,indent=2)


def main(_):
    generate_schema_with_action()


if __name__ == "__main__":
    flags.mark_flag_as_required("sgd_folder")
    flags.mark_flag_as_required("schema_file")
    flags.mark_flag_as_required("out_schema_file")
    app.run(main)