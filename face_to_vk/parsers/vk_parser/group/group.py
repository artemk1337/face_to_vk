from settings import PARSER_ALL


class GroupIdsParser:

    MAX_VALUE = 1000  # error if more
    METHOD_GROUPS = 'groups.getMembers'

    def __init__(self, group_id):
        self.group_id = group_id

    def parse_all(self) -> list:
        """
        >>> GroupIdsParser(139105204).parse_all()

        """
        users = PARSER_ALL(
            self.METHOD_GROUPS,
            self.MAX_VALUE,
            values={'group_id': self.group_id, 'sort': 'id_asc'}
        )
        return users['items']
