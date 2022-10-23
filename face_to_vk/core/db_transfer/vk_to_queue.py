import uuid

from core.db_connector.commands.queue import QueueConnector
from core.vk_parser import VKParseMethods
from settings import MAIN_LOGGER as LOGGER


class VkToQueue:

    def __init__(self):
        self.my_uuid = str(uuid.uuid4())

    def _save_db(self, user_ids: list):
        QueueConnector.insert(user_ids=user_ids, uuid=self.my_uuid)

    def parse_ids_from_group_to_db(self, group_id: int):
        """
        Parse user ids and save in DB
        :param group_id:
        :return:
        >>> VkToQueue().parse_ids_from_group_to_db(182710778)
        """
        LOGGER.info("Parsing user ids...")
        user_ids = VKParseMethods.parse_user_ids_from_group(group_id)
        self._save_db(user_ids=user_ids)
        LOGGER.info("Success parsed user ids")

    def parse_ids_from_search_to_db(self, q: str, **kwargs):
        """
        Parse user ids from search
        :param q: string
        :param kwargs:
            {sort: int = 0,
            sex: Optional[int],
            country: Optional[int],
            city: Optional[int],
            home_town: Optional[int],
            university_country: Optional[int],
            age_from: Optional[int],
            age_to: Optional[int],
            online: Optional[int],
            has_photo: Optional[int],
            from_list: Optional[str]}
        """
        LOGGER.info("Parsing user ids...")
        user_ids = VKParseMethods.parse_ids_from_search(q, **kwargs)
        self._save_db(user_ids=user_ids)
        LOGGER.info("Success parsed user ids")
