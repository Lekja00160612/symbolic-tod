# pytype: skip-file
r"""Create text format SGD data for generative models.

This is symbolic version, which generates the data format:
"""

import copy
import json
import os
import pathlib
import random
import string
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
from absl import app
from absl import flags
from absl import logging

from utils import PARAMS_PREFIX, SYSTEM_ACTION_PREFIX, USER_ACTION_PREFIX
from utils.symbolize import symbolize
from utils.action_template import (
    ActionTemplate,
    OODTemplate,
    NON_DOMAIN_ACTIONS,
    resolve_multi_domain_action,
)

# from utils.action_template import (
#     NON_SLOT_CONTEXT_ACTIONS,
#     SLOT_CONTEXT_ACTIONS,
#     DEPENDENCIES_ACTIONS,
#     SYSTEM_ELIMINATE_ACTIONS,
# )
from utils.action_template import *
from utils.schema_info import SchemaInfo, load_schema
from utils.turn_info import TurnInfo
from utils.helper import (
    merge_domain_name,
    try_lowercase,
    try_shuffle,
    is_in_domain,
    add_string,
    get_name,
    try_sort_id,
)
import utils.flags  # pylint: disable=unused-import

logging.set_verbosity(logging.INFO)
FLAGS = flags.FLAGS

MISSING_FORMAT = "{name} id not found in {desc_to_id}"


class TurnProcessor:
    """
    These turns need process input and output, use serialize for this task
    """
    def __init__(
        self,
        speaker: str,
        turn_id: int,
        dialogue_id: int,
        frames,
        domain_frame,
        item_desc: SchemaInfo,
        state_dict,
        desc_to_id,
    ):
        self.speaker = speaker
        self.turn_id = turn_id
        self.dialogue_id = dialogue_id
        # Save labeled state
        self.frames = frames
        self.frame = domain_frame
        if speaker == "user":
            self.state = self.frame["state"]
        # Name of currenty process domain
        self.domain = try_lowercase(self.frame["service"])

        # Save information from schema
        self.item_desc = item_desc

        # Save information that need to aggregate among turns
        self.state_dict = state_dict
        # Internal usage for description to id
        self.desc_to_id = desc_to_id

    def process_params(self) -> Tuple[str, Dict[str, str]]:
        """
        Process <params> tag for each turn
        Rule:
            - Each <param> is then mapped to format "<param_id>=<param_desc>"
        Return:
            - str - params_desc: Contains all formatted params, then append to <tag>
            - Dict - param_to_id: For further processing <acts>, we need to map description to id
        """
        # Init dictionary mapping from "number_of_people" to "p69"
        param_to_id = {}
        param_desc = str()
        params = list(self.item_desc.params.keys())
        try_shuffle(params)
        for param_id, param in enumerate(params):
            if not is_in_domain(self.domain, param):
                continue

            param_name = get_name(param)
            match FLAGS.data_format:
                case "full_desc":
                    desc = self.item_desc.params[param]
                case "item_name":
                    desc = param_name

            # If we are generating with multiple choice
            # and param is categorical, append choices.
            # Example: 3a=balance 3b=credit
            if FLAGS.multiple_choice != "none" and self.item_desc.is_categorical[param]:
                possible_values = self.item_desc.possible_values[param]
                try_shuffle(possible_values)
                assert len(possible_values) < len(string.ascii_lowercase)
                letters = list(string.ascii_lowercase)

                possible_values_pieces = []
                for letter, value in zip(letters, possible_values):
                    if FLAGS.multiple_choice == "1a":
                        possible_values_pieces.append(f"{param_id}{letter}) {value}")
                    elif FLAGS.multiple_choice == "a":
                        possible_values_pieces.append(f"{letter}) {value}")
                desc = add_string(desc, " ".join(possible_values_pieces))

            _id = f"{PARAMS_PREFIX}{param_id}"
            desc = add_string(_id, desc, delimiter=FLAGS.delimiter)
            param_desc = add_string(param_desc, desc, delimiter="; ")

            self.desc_to_id[param] = _id
            param_to_id[param_name] = _id

        logging.debug(param_desc)
        return param_desc, param_to_id

    def process_actions(self, speaker: str, param_to_id: Dict[str, str]) -> str:
        """
        Process <acts> tag for each turn given <speaker> and <param_to_id>
        Rule:
            - Actions to process = Possible actions + OOD actions
              OOD_actions for user: inform, request, for system: inform, request, confirm
              OOD is then used to switch domains and is highly related to <conversation>
            - Each <act> is then mapped to format "<atc_id>=<act_desc>"
        Return:
            - str - acts_desc: Contains all formatted actions, then append to <tag>
        """

        match speaker:
            case "user":
                actions = list(self.item_desc.possible_user_actions.keys())
                ood_actions = USER_OOD_ACTIONS
                possible_actions = self.item_desc.possible_user_actions
                id_prefix = USER_ACTION_PREFIX
            case "system":
                actions = list(self.item_desc.possible_system_actions.keys())
                ood_actions = SYSTEM_OOD_ACTIONS
                possible_actions = self.item_desc.possible_system_actions
                id_prefix = SYSTEM_ACTION_PREFIX
            case _:
                logging.fatal(f"Unavailble speaker: {speaker}")

        # Later OOD action ids are randomly picked from empty_ids
        empty_ids = []

        action_desc = list()
        try_shuffle(actions)

        for action_id, action in enumerate(actions):
            if not is_in_domain(self.domain, action):
                empty_ids.append(action_id)
                continue

            match FLAGS.data_format:
                case "full_desc":
                    desc = possible_actions[action]
                    # Checking if action need to resolve param_id
                    if (
                        # if symbolize_level include symbolize action
                        "action" in FLAGS.symbolize_level
                        # and action is in one of SLOT_CONTEXT_ACTIONS
                        and any(
                            slot_context_action in action
                            for slot_context_action in SLOT_CONTEXT_ACTIONS
                        )
                        # and not in one of NON_SLOT_CONTEXT_ACTIONS
                        and (
                            not any(
                                non_slot_context_action in action
                                for non_slot_context_action in NON_SLOT_CONTEXT_ACTIONS
                            )
                        )
                    ):
                        for param_name in param_to_id.keys():
                            if param_name in desc:
                                logging.debug(
                                    f"{param_name} is replaced with {param_to_id[param_name]}"
                                )
                                desc = desc.replace(param_name, param_to_id[param_name])
                                break
                        # Log error if action need <param> but cannot resolve <param_id>
                        else:
                            logging.fatal(
                                f"{action}, {(list((non_slot_context_action in desc, non_slot_context_action) for non_slot_context_action in NON_SLOT_CONTEXT_ACTIONS))}\n{MISSING_FORMAT.format(name=action,desc_to_id=param_to_id)}"
                            )
                case "item_name":
                    desc = get_name(action)

            _id = f"{id_prefix}{action_id}"
            self.desc_to_id[action] = _id
            desc = add_string(_id, desc, delimiter=FLAGS.delimiter)
            action_desc.append((_id, desc))

        # Foreach ood actions, take random empty index and add to action_ids list
        # then sort action_ids list
        for ood_action, ood_desc in ood_actions:
            action_id = random.choice(empty_ids)
            empty_ids.remove(action_id)
            _id = f"{id_prefix}{action_id}"
            self.desc_to_id[ood_action] = _id
            desc = add_string(_id, ood_desc, delimiter=FLAGS.delimiter)
            action_desc.append((_id, desc))

        try_sort_id(action_desc)
        action_desc = "; ".join(x[1] for x in action_desc)
        logging.debug(action_desc)
        return action_desc

    def process_dependencies(
        self,
    ) -> str:
        """
        Process <dependencies> tag for each turn
        Rule:
            - Dependency means what actions needed to happen before database query
            - Each <dependency> (query_act) is then mapped to format:
              "<previous_act_id_1, previous_act_id_2,... previous_act_id_n> -> <query_act_id>"
            # - It is unclear what actions must proceed before query, however, <required slots> are.
            #   Foreach required_slot, we try:
            #     1. user_inform_<required_slot>
            #     2. system_offer_<required_slot>
            #     3. system_confirm_<required_slot>, Only happen if the action is transactional
            #   (1), (2) exist because <required_slot> must come from either system offer or user inform
            #   (3) exists because transactional query need confirmation before proceed
            - During dataset testing only, if the intent not is_transactional (think of http GET method)
              we default all its required slots are from user_inform, else, we must system_confirm
              if the intent is_transactional (think of http POST method)
        Return:
            - str - dependencies_desc: Contains all formatted dependencies, then append to <tag>
        """
        dependencies = list(self.item_desc.dependencies.keys())
        try_shuffle(dependencies)
        dependency_desc = []
        for dependency in dependencies:
            if not is_in_domain(self.domain, dependency):
                continue

            depend_variable_ids = []
            # dependency_actions = DEPENDENCIES_ACTIONS
            # is transactional intent
            if self.item_desc.is_transactional[dependency] is True:
                dependency_actions = (
                    # DEPENDENCIES_ACTIONS + 
                    TRANSACTIONAL_DEPENDENCIES_ACTIONS
                )
            # not transactional intent
            else:
                dependency_actions = (
                    DEPENDENCIES_ACTIONS
                )

            for required_slot in self.item_desc.dependencies[dependency]["required_slots"]:
                actions = []
                for dependency_action in dependency_actions:
                    action_name = f"{self.domain}-{dependency_action}_{required_slot}"
                    actions.append(action_name)
                    # If not getting the id, this bahavior is normal as DEPENDENCIES_ACTIONS
                    # only contain suspicious actions, not always appear in possible_actions
                    _id = self.desc_to_id.get(action_name, None)
                    if _id is not None:
                        depend_variable_ids.append(_id)
                # Must be atleast one action matches the context
                if len(depend_variable_ids) == 0:
                    logging.fatal(
                        f"Non-action exist in context: {actions} not found {MISSING_FORMAT.format(name=action_name, desc_to_id=self.desc_to_id)}"
                    )
            try_sort_id(depend_variable_ids, based_index=-1)
            desc = f"{', '.join(depend_variable_ids)} -> {self.desc_to_id[dependency]}"
            dependency_desc.append(desc)
        dependency_desc = "; ".join(dependency_desc)
        logging.debug(dependency_desc)
        return dependency_desc

    def process_target_acts(
        self,
    ) -> str:
        """
        Process <targetacts> tag for each turn
        Rule:
            - targetacts is defined as query_action_id of current active intent,
              in this dataset, only a action is activated at one time
            - each targetact is formatted as "<targetact_id>", there might be turns have no target actions, <targetacts> = NONE_ACT
        Return:
            - str - targetacts_desc: Contains all formatted target_actions, then append to <tag>
        """
        intent = try_lowercase(self.state["active_intent"])
        # No active intent
        target_desc = str()
        if intent == "none":
            target_desc += NONE_ACT
        # Available active intent
        else:
            intent = merge_domain_name(
                self.domain,
                f"system_{ActionTemplate.KEY_SYSTEM_QUERY.format(intent_name=intent)}",
            )
            # Lookup for intent action id
            try:
                _id = self.desc_to_id[intent]
            except Exception:
                logging.fatal(
                    MISSING_FORMAT.format(name=intent, desc_to_id=self.desc_to_id)
                )
            target_desc += _id
        logging.debug(target_desc)
        return target_desc

    def process_constraints(
        self,
    ) -> str:
        """
        Process <contraints> tag for each turn
        Rule:
            - contraint is defined as a natural language commonsense of human worker,
              that we want the model to mimic
            - each contraint is formatted as "<constraint_desc>"
        Return:
            - str - constraints_desc: Contains all formatted target_actions, then append to <tag>
        """
        constraints = [
            "user request param x, system inform the param x",
            "target actions depend on param x and user did not inform, system request param x",
        ]
        try_shuffle(constraints)
        constrain_desc = "; ".join(constraints)
        logging.debug(constrain_desc)
        return constrain_desc

    def process_state_and_conversation(
        self,
    ) -> Tuple[str, str]:
        """
        Process <state> tag and <conversation> tag for each turn
        Rule:
            - we may symbolize <param_value> and then change all corresponding value in the <conversation>

            For <state>:
            - <state> contains params and their corresponding values from the conversation
            - Each param has format "<param_id>=<param_value>"
              To be qualify for symbolize, the slot must be non-categorical
            For <conversation>:
            - <conversation> contains all history utterances
            - Each utterance is formatted as "[<speaker>] <utterance>"
        Return:
            - str - state_desc: Contains all formatted state params, then append to <tag>
            - str - conversation_desc: Contains all formatted history utterance
        """
        desc = []
        for slot, slot_values in self.state["slot_values"].items():
            slot = merge_domain_name(self.domain, slot)
            slot_value = try_lowercase(slot_values[0])
            try:
                _id = self.desc_to_id[slot]
                is_categorical = self.item_desc.is_categorical[slot]
            except Exception:
                logging.fatal(
                    MISSING_FORMAT.format(name=slot, desc_to_id=self.desc_to_id)
                )
            desc.append((_id, slot_value, is_categorical))

        try_sort_id(desc)
        conversation = self.state_dict["conversation"]

        if "value" in str(FLAGS.symbolize_level):
            # Symbolize value and replace that value with a generated symbol in conversation
            for index, param in enumerate(desc):
                param_id, param_value, param_is_categorical = param
                # if param is categorical and param_symbolize_level is non-categorical
                if (
                    param_is_categorical
                    and FLAGS.param_symbolize_level == "non-categorical"
                ):
                    continue
                symbolize_value = symbolize(param_value)
                desc[index] = (param_id, symbolize_value)
                logging.debug(f"{param_value} is replaced with {symbolize_value}")

                def _replace_symbolic(turn_utterance):
                    # pylint: disable=no-member
                    speaker, utterance = turn_utterance
                    if param_value in utterance and param_value != symbolize_value:
                        utterance = utterance.replace(param_value, symbolize_value)
                    # pylint: enable=no-member
                    return [speaker, utterance]

                conversation = list(map(_replace_symbolic, conversation))
        conversation = list(map(lambda x: f"[{x[0]}] {x[1]}", conversation))
        conversation_desc = " ".join(conversation)
        state_desc = " ".join(map(lambda x: f"{x[0]}={x[1]}", desc))
        return state_desc, conversation_desc

    def process_next_actions_and_history(
        self,
        previous_domain,
    ) -> Tuple[Union[List, str], Union[List, str]]:
        """
        Process <history> tag and <nextacts> tag for each turn given previous domain. Note these tags are complicated
        Rule:
            - we may symbolize <param_value> and then change all corresponding value in the <conversation>

            For <history>:
            - <history> contains action_ids for every turn from the conversation
            - <history> of each turn is formatted as "<action_id_1>, <action_id_2>, ... <action_id_n>" and seperated by ";"
            - Single-domain:
                - When history only contain a domain, no complex logic is required
            - Multi-domain:
                - The previous domain actions are now not available, mapping history actions to their OOD actions is required
                  Except for these actions, all other actions are kept unchanged:
                    1. inform_<slot>, inform_intent -> inform_undefined
                    2. request_<slot> -> request_undefined
                    3. system_confirm_<slot> -> system_confirm_undefined
                    4. All query action from previous domains are removed as the schema no longer signal the model when to query
            For <nextacts>:
            - <nextacts> contains action_ids for the next turn, harvest during process system turn
              System-turn's actions is previous user-turn next_actions
            - If the next actions list contain any "offer_<slot>", "notify_<failure/success>", "inform_count"
              remove those actions and replace by query the active intent as in practice, we do not know the
              query result yet during this stage, we only know we must query
            - Remove "offer_intent" as we want to let dialogue manager to decide what to do next, afterall, the next actions
              are just for recommending
            - Each actions is formatted as "<action_id>", seperate by ","
            - There are turns that have no actions due to domain transition, if then <nextacts> = NONE_ACT
            - During transition turn, new domain actions will not be recognize
        Return:
            if user_turn:
            - str - history_desc: Contains all formatted history actions
            - [] - next_actions:
            if system_turn:
            - [] - history_desc
            - str - nextacts_desc: Contains all formatted next actions or none
        """
        # Handle last turn actions for multiple-domains in user turn
        # len start at 1 while turn_id start at 0, turn_id must be 1 addition ahead
        if len(self.state_dict["history"]) <= self.turn_id:
            self.state_dict["history"].append(defaultdict(lambda: {}))
            self.state_dict["active_intent"].append(defaultdict(lambda: {}))
            for frame in self.frames:
                domain_next_actions = []
                domain = try_lowercase(frame["service"])
                for action in frame["actions"]:
                    act = try_lowercase(action["act"])
                    slot = try_lowercase(action["slot"])
                    # No slot action (greeting, select, ...)
                    if slot == "" or slot == "count":
                        action = f"{self.speaker}_{act}"
                    # Intent related action
                    elif slot == "intent":
                        value = (
                            try_lowercase(action["values"][0])
                            if slot == "intent"
                            else None
                        )
                        action = f"{self.speaker}_{act}_{value}"
                    # Slot required action (inform, request, ...)
                    else:
                        action = f"{self.speaker}_{act}_{slot}"

                    # Resolve special action: affirm_intent, negate_intent
                    if "affirm_intent" in action:
                        action = action.replace("affirm_intent", "affirm")
                    elif "negate_intent" in action:
                        action = action.replace("negate_intent", "negate")
                    domain_next_actions.append(
                        try_lowercase(merge_domain_name(domain, action))
                    )
                if self.speaker == "user":
                    self.state_dict["active_intent"][-1][domain] = try_lowercase(
                        frame["state"]["active_intent"]
                    )
                self.state_dict["history"][-1][domain] = domain_next_actions

        logging.debug(
            f"{self.dialogue_id}, {self.turn_id}: {self.state_dict['history']=}"
        )

        # History and next actions are 2 special tags that need careful design to handle
        # take example `events_2-user_request_alts` during `buses_2` domain will fail,
        # therefore all actions which do not use domain must first be parsed from domain
        non_domain_desc_to_id = {}
        for action in self.desc_to_id.keys():
            non_domain_action = get_name(action)
            if non_domain_action in NON_DOMAIN_ACTIONS:
                non_domain_desc_to_id[non_domain_action] = self.desc_to_id[action]

        self.desc_to_id.update(non_domain_desc_to_id)

        # Handle next actions
        next_actions_desc = list()
        if self.speaker == "system":
            # We know there is only one frame for the last history turn (system turn)
            for domain, actions_list in self.state_dict["history"][-1].items():
                next_actions_desc += actions_list
            # During service call, system will offer multiple slots,
            # however we want our system to stop at service calling and nothing further
            # therefore it is necessary to eliminate all offer actions within this turn system actions
            # and replace them with only one action - db call.
            previous_action_length = len(next_actions_desc)
            next_actions_desc = [
                action
                for action in next_actions_desc
                if (
                    # eliminate if action in any of "offer", "notify", "notify_success", "notify_failure", "inform_count"
                    # "offer_intent" is a controversial decision, should dialogue manager take care of this or the state tracker?
                    # for the sake of sanity, in this experiment, we do not keep "offer_intent" in next_action
                    not any(
                        eliminate_action in action
                        for eliminate_action in NEXTACTS_ELIMINATE_ACTIONS_QUERY_RELATED
                    )
                    # keep the action if the action is "offer_intent", we will remove it latter
                    or any(
                        exclude_eliminate_action in action
                        for exclude_eliminate_action in NEXTACTS_ELIMINATE_ACTIONS
                    )
                )
            ]
            current_action_length = len(next_actions_desc)

            # There are some actions eliminated, add query into action
            if current_action_length < previous_action_length:
                service_call = self.frame.get("service_call", None)
                # if there is no service call in current system turn,
                # assume to call the last user turn active intent
                method_name = (
                    try_lowercase(self.frame["service_call"]["method"])
                    if service_call
                    else self.state_dict["active_intent"][-2][
                        domain
                    ]  # index -1 is current_turn which has no active_intent
                )
                query_action = merge_domain_name(
                    self.domain,
                    f"system_{ActionTemplate.KEY_SYSTEM_QUERY.format(intent_name=method_name)}",
                )
                next_actions_desc.append(query_action)
                self.state_dict["history"][-1][self.domain].append(query_action)

            action_ids = list()
            next_actions_desc = [
                action
                for action in next_actions_desc
                if (
                    # remove the action if the action is "offer_intent", we want dialogue manager to decide this
                    not any(
                        exclude_eliminate_action in action
                        for exclude_eliminate_action in NEXTACTS_ELIMINATE_ACTIONS
                    )
                )
            ]
            # Resolve actions only if it matched with previous_domain
            # LONG EXPLAIN
            # The reason behind this is that only user has the right to transit domain
            # If user asked for domain transition, the user turn would has 2 domains,
            # each will then correspond to a system nextacts process
            # If domain is differ from previous domain then it means the actions should be ignore since it has yet existed
            # SHORT EXPLAIN
            # During transition, previous user turn has at least 1 turn differ from current system turn domain
            # When continue to process that previous turn, we as a model must not know what to do until new domain is detected
            for action in next_actions_desc:
                if is_in_domain(self.domain, action) and self.domain == previous_domain:
                    try:
                        _id = self.desc_to_id[action]
                        action_ids.append(_id)
                    except Exception:
                        logging.fatal(
                            f"{self.dialogue_id}, {self.turn_id}, {self.domain}, {previous_domain}: "
                            + MISSING_FORMAT.format(
                                name=action, desc_to_id=self.desc_to_id
                            )
                        )

            # If action_ids is empty next actions is none
            if action_ids:
                try_sort_id(action_ids, based_index=-1)
                next_actions_desc = ", ".join(action_ids)
            # If action_ids is empty, the domain transition is taking place
            else:
                next_actions_desc = NONE_ACT

        history_desc = list()
        # Handle history
        if self.speaker == "user":
            for turn in self.state_dict["history"]:
                action_ids = []
                for domain, previous_actions_list in turn.items():
                    # Handle out of domain actions
                    actions_list = []
                    if domain != self.domain:
                        for action in previous_actions_list:
                            speaker = "user" if "user" in action else "system"
                            action = resolve_multi_domain_action(
                                speaker=speaker, action=action
                            )
                            # eliminate query actions
                            if action is not None:
                                actions_list.append(action)
                    else:
                        actions_list = previous_actions_list

                    for action in actions_list:
                        try:
                            _id = self.desc_to_id[action]
                            action_ids.append(_id)
                        except Exception:
                            logging.fatal(
                                f"{self.dialogue_id}, {self.turn_id}, {self.domain}, {previous_domain}: "
                                + MISSING_FORMAT.format(
                                    name=action, desc_to_id=self.desc_to_id
                                )
                            )
                try_sort_id(action_ids, based_index=-1)
                desc = ", ".join(action_ids)
                history_desc.append(desc)
            history_desc = "; ".join(history_desc)

            logging.debug(f"{next_actions_desc}")
            logging.debug(f"{history_desc}")
        return next_actions_desc, history_desc

    def serialize(self, turn_info: TurnInfo) -> Tuple[TurnInfo, Dict[str, str]]:
        if self.speaker == "user":
            params_desc, param_to_id = self.process_params()
            turn_info.in_params = add_string(
                turn_info.in_params, params_desc, first_empty=True
            )
            user_actions_desc = self.process_actions(
                speaker="user", param_to_id=param_to_id
            )
            turn_info.in_user_actions = add_string(
                turn_info.in_user_actions, user_actions_desc
            )
            system_actions_desc = self.process_actions(
                speaker="system", param_to_id=param_to_id
            )
            turn_info.in_system_actions = add_string(
                turn_info.in_system_actions, system_actions_desc
            )
            next_actions_desc, history_desc = self.process_next_actions_and_history(
                turn_info.frame_domain
            )
            turn_info.out_history = add_string(turn_info.out_history, history_desc)
            dependencies_desc = self.process_dependencies()
            turn_info.in_dependencies = add_string(
                turn_info.in_dependencies, dependencies_desc
            )
            target_acts_desc = self.process_target_acts()
            turn_info.in_target_actions = add_string(
                turn_info.in_target_actions, target_acts_desc
            )
            constraints_desc = self.process_constraints()
            turn_info.in_constraints = add_string(
                turn_info.in_constraints, constraints_desc
            )
            state_desc, conversation_desc = self.process_state_and_conversation()
            turn_info.out_states = add_string(turn_info.out_states, state_desc)
            turn_info.in_conversations = add_string(
                turn_info.in_conversations, conversation_desc
            )

        elif self.speaker == "system":
            next_actions_desc, history_desc = self.process_next_actions_and_history(
                previous_domain=turn_info.frame_domain
            )
            turn_info.out_next_actions = add_string(
                turn_info.out_next_actions, next_actions_desc
            )
        else:
            logging.fatal("Error checking conditions!")

        return turn_info, self.desc_to_id


def process_turn(
    dialogue_id: int,
    turn_id: int,
    turn: Dict[str, Any],
    item_desc: SchemaInfo,
    state_dict: Dict[str, str],
) -> str:
    """Collects information from a single turn.

    Args:
      turn: A dictionary containing original turn structure.
      turn_info: A dictionary accmulating essential info from each turn.
      cumu_slots: An OrderedDict containing cumumulative slot information.
      item_desc: A dictionary of scheam items and their descriptions.
      prefix: A string of the schema item description prefix.
      turn_id: Integer index of turn in dialogue.

    Returns:
      Prefix string (item descriptions) from the current turn and per-frame
      TurnInfo objects.
    """
    speaker = try_lowercase(turn["speaker"])
    utterance = try_lowercase(turn["utterance"])
    state_dict["conversation"].append((speaker, utterance))

    turn_info_per_frame = []

    # Examine multi domain example during data exploration
    # if len(turn["frames"]) > 1 and turn["speaker"] == "system":
    #     logging.fatal("System failed!")
    #     acts = []
    #     for frame in turn['frames']:
    #         [acts.append(action['act'].lower()) for action in frame['actions']]
    #     # Check whether `inform_intent` exist in actions of any frame
    #     if "inform_intent" not in acts or len(turn["frames"]) > 2: # or True:
    #         logging.warning(f"{turn_info.dialogue_id} {turn_id}")
    #         logging.warning(f"{state_dict['conversation'][-2:]}")
    #         for frame in turn['frames']:
    #             logging.warning(f"<{frame['service']}>: {[action['act'] for action in frame['actions']]}")
    # return {}, []

    if speaker == "user":
        state_dict["last_turn_infos"] = {}
        for frame_id, frame in enumerate(turn["frames"]):
            turn_info = TurnInfo()
            turn_info.user_turn = True
            turn_info.turn_id = str(turn_id)
            turn_info.dialogue_id = str(dialogue_id)

            # Multi-service turns are possible, each frame corresponds to one
            # service (domain).
            turn_info.frame_id = str(frame_id)
            turn_info.frame_domain = frame["service"].lower()

            turn_info, desc_to_id = TurnProcessor(
                speaker="user",
                turn_id=turn_id,
                dialogue_id=dialogue_id,
                frames=turn["frames"],
                domain_frame=frame,
                item_desc=item_desc,
                state_dict=state_dict,
                desc_to_id={},
            ).serialize(turn_info=turn_info)
            turn_info_per_frame.append(copy.deepcopy(turn_info))

            state_dict["last_turn_infos"][turn_info.frame_domain] = (
                copy.deepcopy(turn_info),
                copy.deepcopy(desc_to_id),
            )
    elif speaker == "system":
        assert len(turn["frames"]) == 1, "System turn must has exactly 1 frame only"
        frame = turn["frames"][0]
        # we dont use domain as domain is also recorded in turn_info.frame_domain
        for domain, last_turn_info in state_dict["last_turn_infos"].items():
            turn_info, desc_to_id = last_turn_info
            turn_info.user_turn = False
            turn_info.turn_id = str(turn_id)

            turn_info, desc_to_id = TurnProcessor(
                speaker="system",
                turn_id=turn_id,
                dialogue_id=dialogue_id,
                frames=turn["frames"],
                domain_frame=frame,
                item_desc=item_desc,
                state_dict=state_dict,
                desc_to_id=desc_to_id,
            ).serialize(turn_info=turn_info)
            turn_info_per_frame.append(copy.deepcopy(turn_info))

    return desc_to_id, turn_info_per_frame


def write_examples(turn_list: List[TurnInfo], out_file) -> None:
    """Format output example strings and write to file.

    Args:
      turn_list: A list of dict accmulating essential info from each turn.
      out_file: A GFile object for file output.
    """
    for turn_info in turn_list:
        # Write samples to file. Each example is divided into three parts - input, output, meta
        # separated by \t\t, each components are spaced by \t, the first part being inputs to the model, and the
        # second part are labels for prediction.
        input_text = ""
        output_text = ""
        if FLAGS.level == "dst":
            if not turn_info.user_turn:
                # Only output at user turns.
                input_text = f"""{turn_info.in_params}\t{turn_info.in_user_actions}\t{turn_info.in_system_actions}\t\
{turn_info.in_dependencies}\t{turn_info.in_target_actions}\t{turn_info.in_constraints}\t\
{turn_info.in_conversations}"""
                output_text = f"{turn_info.out_states}\t{turn_info.out_history}\t{turn_info.out_next_actions}"
        if input_text != "" and output_text != "":
            # Add dialogue ID, turn ID and frame ID to the target for later eval.
            # Occasionally some examples include newline in the middle.
            example = (
                f"{input_text}\t\t{output_text}\t\t"
                + f"dialogue_id:{turn_info.dialogue_id}\tturn_id:{turn_info.turn_id}\tframe_domain:{turn_info.frame_domain}\tframe_id:{turn_info.frame_id}"
            )
            if FLAGS.lowercase:
                example = example.lower()

            logging.debug(example)
            out_file.write(f"{example}\n")


def example_filter(turn_list: List[TurnInfo]):
    """Extract specified percentage of examples.

    And ensure uniform domain distribution if specified.

    Args:
      turn_list: A list of TurnInfo containing all examples.

    Returns:
      Specified percentage of examples, with uniform domain distribution if
      needed.
    """
    if FLAGS.data_percent == 0.0:
        return turn_list

    out_sample_num = int(len(turn_list) * FLAGS.data_percent)
    if not FLAGS.uniform_domain_distribution:
        try_shuffle(turn_list)
        return turn_list[:out_sample_num]
    else:
        domain_examples = {}
        domain_id = {}
        domain_count = 0
        for turn in turn_list:
            if turn.turn_domain in domain_id:
                domain_examples[domain_id[turn.turn_domain]].append(turn)
            else:
                domain_examples[domain_count] = [turn]
                domain_id[turn.turn_domain] = domain_count
                domain_count += 1

        # How many examples from each domain has been added to the final list.
        consumed_examples = {d: 0 for d in range(domain_count)}
        uniform_turn_list = []
        for s in range(out_sample_num):
            # Find first domain that still has unused examples.
            domain_id = s % domain_count
            for d in range(domain_count):
                cand_domain = (domain_id + d) % domain_count
                if len(domain_examples[cand_domain]) > consumed_examples[cand_domain]:
                    domain_id = cand_domain
                    break

            uniform_turn_list.append(
                domain_examples[domain_id][consumed_examples[domain_id]]
            )
            consumed_examples[domain_id] += 1

        try_shuffle(uniform_turn_list)

        return uniform_turn_list


def preprocess_dialogues(dialogues):
    for dialogue_index, dialogue in enumerate(dialogues):
        speaker = str()
        previous_domains = set()
        current_domains = set()
        for turn_index, turn in enumerate(dialogue["turns"]):
            previous_domains = current_domains
            current_domains = set()
            speaker = try_lowercase(turn["speaker"])
            for frame in turn["frames"]:
                current_domains.add(try_lowercase(frame["service"]))
            if current_domains == previous_domains:
                continue
            # domain transition occuring
            else:
                if speaker == "system":
                    # previous_speaker is user, user must have informed some intent,
                    # this is normal case and system will follow user intent
                    if not current_domains.issubset(previous_domains):
                        logging.fatal(
                            f"System changed the domain by itself!!!\n\t\t\t{dialogue['dialogue_id']}, {turn_index}, user domains: {previous_domains}, system domains: {current_domains}"
                        )
                elif speaker == "user":
                    for domain in previous_domains:
                        # If not the first turn and previous system turn has some domain that current user turn doesn't
                        # Add that domain to user turn as examples will based on user turn
                        if domain not in current_domains and turn_index != 0:
                            current_domains.add(domain)
                            padding_frame = {
                                "actions": [],
                                "service": domain,
                                "slots": [],
                                "state": {
                                    "active_intent": "NONE",
                                    "requested_slots": [],
                                    "slot_values": {},
                                },
                            }
                            dialogues[dialogue_index]["turns"][turn_index][
                                "frames"
                            ].append(padding_frame)
                            logging.info(
                                f"{dialogue['dialogue_id']}, {turn_index}, system domains: {previous_domains}, user domains: {current_domains}"
                            )


def generate_data(item_desc):
    """Generate SGD examples in text format.

    Args:
      ordered_slots: An ordered dictionary containing slot names.
      item_desc: A dictionary containing items and their descriptions.
    """
    if not os.path.isdir(os.path.dirname(FLAGS.output_file)):
        os.makedirs(os.path.dirname(FLAGS.output_file))
    with open(FLAGS.output_file, "w", encoding="utf-8") as out_file:
        all_turns_per_frame = []

        sgd_folder = pathlib.Path(FLAGS.sgd_file)
        for sgd_file in sgd_folder.rglob("dialogues_*.json"):
            logging.info(f"processing {sgd_file}")
            if not any([str(number) in str(sgd_file) for number in range(44, 128)]):
                continue
            with open(sgd_file, "r", encoding="utf-8") as sgd_in:
                dialogues = json.load(sgd_in)
                preprocess_dialogues(dialogues)
                # logging.info(json.dumps(dialogues, indent=4))
                for dialogue in dialogues:
                    state_dict = {
                        "history": [],
                        "conversation": [],
                        "last_turn_infos": {},
                        "active_intent": [],
                    }

                    # # Check maximum number of services
                    # services = dlg["services"]
                    # if len(services) > 2:
                    #     logging.warning(f"{turn_info.dialogue_id} has {len(services)} services ({services})")
                    for turn_idx, turn in enumerate(dialogue["turns"]):
                        desc_to_id, per_frame_turn_info = process_turn(
                            dialogue_id=dialogue["dialogue_id"],
                            turn_id=turn_idx,
                            item_desc=item_desc,
                            state_dict=state_dict,
                            turn=turn,
                        )
                        logging.debug(per_frame_turn_info)
                        all_turns_per_frame.extend(copy.deepcopy(per_frame_turn_info))

                write_examples(example_filter(all_turns_per_frame), out_file)


def main(_):
    item_desc = load_schema()
    generate_data(item_desc)


if __name__ == "__main__":
    if not os.path.exists("./log"):
        os.makedirs("./log")
    logging.get_absl_handler().use_absl_log_file("absl_logging", "./log")
    flags.mark_flag_as_required("sgd_file")
    flags.mark_flag_as_required("schema_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
