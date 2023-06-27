""" Template for user and system action """
from enum import StrEnum

class ActionTemplate(StrEnum):
    """
        Default template for all possible action within a dialog
    """
    QUERY_NAME = "query_{intent_name}"

    USER_INFORM = "user inform {slot_name}"
    USER_REQUEST = "user request {slot_name}"
    USER_INTENT = "user want to {intent_name}"
    USER_AFFIRM = "user agree to the offer" #%
    USER_NEGATE = "user deny the offer" #%
    USER_SELECT = "user select {slot_name}" # index_number or slot_name?
    USER_REQUEST_ALTS = "user request alternative items"
    USER_THANKYOU = "user thank"
    USER_GOODBYE = "user goodbye"

    SYSTEM_INFORM = "inform {slot_name}" # to user"
    SYSTEM_REQUEST = "request {slot_name}" # from user"
    SYSTEM_CONFIRM = "ask to confirm value of {slot_name}" 
    SYSTEM_OFFER = "offer user {slot_or_intent_name}" #$
    SYSTEM_NOTIFY_SUCCESS = "notify success" # to user" #$
    SYSTEM_NOTIFY_FAILURE = "notify failure" # to user" #$
    SYSTEM_QUERY = "query {intent_name} API"
    SYSTEM_INFORMCOUNT = "inform number of items satified user"
    SYSTEM_REQMORE = "ask user if they need anything more"
    SYSTEM_GOODBYE = "goodbye user"
