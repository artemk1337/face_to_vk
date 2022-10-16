import uuid

from core.vk_parser import ParseMethods
from core.db_connector.commands.queue import QueueConnector


def vk_search_ids_to_queue(q: str, **kwargs):
    my_uuid = uuid.uuid4()
    ids = ParseMethods.parse_ids_from_search(q, **kwargs)
    QueueConnector.insert(ids, my_uuid)


def vk_groups_ids_to_queue(group_id: int):
    my_uuid = uuid.uuid4()
    ids = ParseMethods.parse_user_ids_from_group(group_id)
    QueueConnector.insert(ids, my_uuid)
