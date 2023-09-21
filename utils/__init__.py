""" Utility for schema guided """
from utils.symbolize import symbolize
from utils.action_template import (
    ActionTemplate,
    resolve_multi_domain_action,
    NON_DOMAIN_SYSTEM_ACTIONS,
    NON_DOMAIN_USER_ACTIONS,
)
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

USER_ACTION_PREFIX = "u"
SYSTEM_ACTION_PREFIX = "s"
PARAMS_PREFIX = "p"
