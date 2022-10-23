from core.vk_parser.base import BaseParser
from ..vk_settings import VK_PARSER_ALL


class GroupIdsParser(BaseParser):
    """
    Parse user ids from group.
    """

    METHOD = 'groups.getMembers'

    def __init__(self, group_id):
        self.group_id = group_id

    def _check_constraints(self):
        pass

    def parse_all(self) -> list:
        """
        Parse all ids.
        :return: list with user ids, ex.: [123, 1234, ...]
        >>> GroupIdsParser(139105204).parse_all()
        """
        users = VK_PARSER_ALL(
            self.METHOD,
            self.MAX_VALUE,
            values={'group_id': self.group_id, 'sort': 'id_asc'}
        )
        return users['items']
