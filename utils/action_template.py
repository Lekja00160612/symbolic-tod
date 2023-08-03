""" Template for user and system action """
from enum import StrEnum
from absl import logging

SLOT_CONTEXT_ACTIONS = ["inform", "request", "confirm", "offer"]
NON_SLOT_CONTEXT_ACTIONS = [
    "inform_intent",
    "request_alts",
    "inform_count",
    "offer_intent",
    "query",
]

DEPENDENCIES_ACTIONS = ["user_inform"]
TRANSACTIONAL_DEPENDENCIES_ACTIONS = ["system_confirm", "system_offer"]

NEXTACTS_ELIMINATE_ACTIONS_QUERY_RELATED = [
    "offer",
    "notify_failure",
    "notify_success",
    "inform_count",
]
NEXTACTS_ELIMINATE_ACTIONS = ["offer_intent"]

NONE_ACT = str()
LOOKUP_INTENTS = ["find", "search", "get"]
PURCHASE_INTENTS = [
    "buy",
    "book",
    "add",
    "reserve",
    "schedule",
    "play",
    "transfer",
    "get",
]


class OODTemplate(StrEnum):
    """
    OOD simulation actions
    """

    KEY_USER_INFORM = "user_inform_undefined"
    KEY_USER_REQUEST = "user_request_undefined"

    KEY_SYSTEM_INFORM = "system_inform_undefined"
    KEY_SYSTEM_REQUEST = "system_request_undefined"
    KEY_SYSTEM_CONFIRM = "system_confirm_undefined"

    USER_INFORM = "user inform undefined information"
    USER_REQUEST = "user request undefined information"

    SYSTEM_INFORM = "inform undefined information"
    SYSTEM_REQUEST = "request undefined information"
    SYSTEM_CONFIRM = "ask to confirm undefined information"


USER_OOD_ACTIONS = [
    (OODTemplate.KEY_USER_INFORM, OODTemplate.USER_INFORM),
    (OODTemplate.KEY_USER_REQUEST, OODTemplate.USER_REQUEST),
]

SYSTEM_OOD_ACTIONS = [
    (OODTemplate.KEY_SYSTEM_CONFIRM, OODTemplate.SYSTEM_CONFIRM),
    (OODTemplate.KEY_SYSTEM_INFORM, OODTemplate.SYSTEM_INFORM),
    (OODTemplate.KEY_SYSTEM_REQUEST, OODTemplate.SYSTEM_REQUEST),
]


class ActionTemplate(StrEnum):
    """
    Default description template for all possible action within a dialog
    """

    KEY_SYSTEM_QUERY = "query_{intent_name}"
    SYSTEM_QUERY = "query {intent_name} API"

    # Note intent_affirm, intent_negate is mapped to affirm, negate respectively
    USER_INFORM = "user inform {slot_name}"
    USER_REQUEST = "user request {slot_name}"
    USER_INTENT = "user want to {intent_name}"
    USER_AFFIRM = "user agree to the offer"
    USER_NEGATE = "user deny the offer"

    USER_SELECT = "user select item"
    # USER_SELECT = "user select {slot_name}"
    USER_REQUEST_ALTS = "user request alternative items"
    USER_THANK_YOU = "user thank"
    USER_GOODBYE = "user goodbye"

    KEY_USER_INFORM = "user_inform_{slot_name}"
    KEY_USER_REQUEST = "user_request_{slot_name}"
    KEY_USER_INTENT = "user_inform_intent_{intent_name}"
    KEY_USER_AFFIRM = "user_affirm"
    KEY_USER_NEGATE = "user_negate"
    KEY_USER_SELECT = "user_select"
    KEY_USER_REQUEST_ALTS = "user_request_alts"
    KEY_USER_THANK_YOU = "user_thank_you"
    KEY_USER_GOODBYE = "user_goodbye"

    SYSTEM_INFORM = "inform {slot_name}"
    SYSTEM_REQUEST = "request {slot_name}"
    SYSTEM_CONFIRM = "ask to confirm value of {slot_name}"
    SYSTEM_OFFER = "offer user {slot_or_intent_name}"
    SYSTEM_NOTIFY_SUCCESS = "notify success"
    SYSTEM_NOTIFY_FAILURE = "notify failure"
    SYSTEM_INFORM_COUNT = "inform number of items satified user"
    SYSTEM_REQ_MORE = "ask user if they need anything more"
    SYSTEM_GOODBYE = "goodbye user"

    KEY_SYSTEM_INFORM = "system_inform_{slot_name}"
    KEY_SYSTEM_REQUEST = "system_request_{slot_name}"
    KEY_SYSTEM_CONFIRM = "system_confirm_{slot_name}"
    KEY_SYSTEM_OFFER = "system_offer_{slot_or_intent_name}"
    KEY_SYSTEM_NOTIFY_SUCCESS = "system_notify_success"
    KEY_SYSTEM_NOTIFY_FAILURE = "system_notify_failure"
    KEY_SYSTEM_INFORM_COUNT = "system_inform_count"
    KEY_SYSTEM_REQ_MORE = "system_req_more"
    KEY_SYSTEM_GOODBYE = "system_goodbye"


class MultiDomainActionTemplate(StrEnum):
    """
    Template for obsolete (old) domain actions
    Notice no action during ood has a slot (since ood is based on slot existence)
    """

    USER_INFORM = OODTemplate.KEY_USER_INFORM
    USER_REQUEST = OODTemplate.KEY_USER_REQUEST
    USER_INTENT = OODTemplate.KEY_USER_INFORM
    USER_AFFIRM = ActionTemplate.KEY_USER_AFFIRM
    USER_NEGATE = ActionTemplate.KEY_USER_NEGATE
    USER_SELECT = ActionTemplate.KEY_USER_SELECT
    USER_REQUEST_ALTS = ActionTemplate.KEY_USER_REQUEST_ALTS
    USER_THANK_YOU = ActionTemplate.KEY_USER_THANK_YOU
    USER_GOODBYE = ActionTemplate.KEY_USER_GOODBYE

    SYSTEM_INFORM = OODTemplate.KEY_SYSTEM_INFORM
    SYSTEM_REQUEST = OODTemplate.KEY_SYSTEM_REQUEST
    SYSTEM_CONFIRM = OODTemplate.KEY_SYSTEM_CONFIRM
    SYSTEM_OFFER = OODTemplate.KEY_SYSTEM_INFORM
    SYSTEM_NOTIFY_SUCCESS = ActionTemplate.KEY_SYSTEM_NOTIFY_SUCCESS
    SYSTEM_NOTIFY_FAILURE = ActionTemplate.KEY_SYSTEM_NOTIFY_FAILURE
    # SYSTEM_QUERY = None # Not available during multi-domain OOD
    SYSTEM_INFORM_COUNT = ActionTemplate.KEY_SYSTEM_INFORM_COUNT
    SYSTEM_REQ_MORE = ActionTemplate.KEY_SYSTEM_REQ_MORE
    SYSTEM_GOODBYE = ActionTemplate.KEY_SYSTEM_GOODBYE


NON_DOMAIN_USER_ACTIONS = [
    # Common user actions
    MultiDomainActionTemplate.USER_AFFIRM,
    MultiDomainActionTemplate.USER_GOODBYE,
    MultiDomainActionTemplate.USER_NEGATE,
    MultiDomainActionTemplate.USER_REQUEST_ALTS,
    MultiDomainActionTemplate.USER_SELECT,
    MultiDomainActionTemplate.USER_THANK_YOU,
]

NON_DOMAIN_SYSTEM_ACTIONS = [
    # Common system actions
    MultiDomainActionTemplate.SYSTEM_NOTIFY_FAILURE,
    MultiDomainActionTemplate.SYSTEM_NOTIFY_SUCCESS,
    MultiDomainActionTemplate.SYSTEM_REQ_MORE,
    MultiDomainActionTemplate.SYSTEM_GOODBYE,
    MultiDomainActionTemplate.SYSTEM_INFORM_COUNT,
]
NON_DOMAIN_ACTIONS = (
    [
        # Undefined actions
        MultiDomainActionTemplate.USER_INFORM,
        MultiDomainActionTemplate.SYSTEM_INFORM,
        MultiDomainActionTemplate.USER_REQUEST,
        MultiDomainActionTemplate.SYSTEM_REQUEST,
        MultiDomainActionTemplate.SYSTEM_CONFIRM,
        MultiDomainActionTemplate.USER_REQUEST_ALTS,
    ]
    + NON_DOMAIN_USER_ACTIONS
    + NON_DOMAIN_SYSTEM_ACTIONS
)


def resolve_multi_domain_action(speaker: str, action: str) -> str:
    """
    return:
        action_name
    """
    ood_action = str()

    if speaker == "user":
        ### NOTE: Ordered
        if "inform_intent" in action:
            ood_action = MultiDomainActionTemplate.USER_INFORM
        elif "request_alts" in action:
            ood_action = MultiDomainActionTemplate.USER_REQUEST_ALTS
        elif "inform" in action:
            ood_action = MultiDomainActionTemplate.USER_INTENT
        elif "request" in action:
            ood_action = MultiDomainActionTemplate.USER_REQUEST
        elif "affirm" in action:  # include affirm_intent
            ood_action = MultiDomainActionTemplate.USER_AFFIRM
        elif "negate" in action:  # include negate_intent
            ood_action = MultiDomainActionTemplate.USER_NEGATE
        elif "select" in action:
            ood_action = MultiDomainActionTemplate.USER_SELECT
        elif "thank_you" in action:
            ood_action = MultiDomainActionTemplate.USER_THANK_YOU
        elif "goodbye" in action:
            ood_action = MultiDomainActionTemplate.USER_GOODBYE
        else:
            logging.fatal(f"{speaker}, {action} is not handled")

    # SYSTEM_QUERY = None # Not available during multi-domain OOD
    if speaker == "system":
        if "inform" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_INFORM
        elif "request" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_REQUEST
        elif "offer" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_OFFER
        elif "confirm" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_CONFIRM
        elif "notify_success" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_NOTIFY_SUCCESS
        elif "notify_failure" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_NOTIFY_FAILURE
        elif "req_more" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_REQ_MORE
        elif "inform_count" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_INFORM_COUNT
        elif "goodbye" in action:
            ood_action = MultiDomainActionTemplate.SYSTEM_GOODBYE
        elif "query" in action:
            ood_action = None
        else:
            logging.fatal(f"{speaker}, {action} is not handled")

    return ood_action
