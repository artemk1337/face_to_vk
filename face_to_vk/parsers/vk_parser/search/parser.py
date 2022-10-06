from typing import Optional

from parsers.vk_parser.search.format import SearchFormat
from parsers.vk_parser.settings import PARSER_ALL
from parsers.vk_parser.base import BaseParser


class SearchIdsParser(BaseParser):
    """
    Parse user ids from search.
    """

    METHOD = 'users.search'

    def __init__(self, **kwargs):
        """
        Available kwargs: 'q', 'sort', 'sex', 'country', 'city', 'home_town', 'university_country',
        'age_from', 'age_to', 'online', 'has_photo', 'from_list'
        """
        # SearchFormat.__dict__
        self.params = SearchFormat(**kwargs).dict()

    def _check_constraints(self):
        pass

    def parse_all(self) -> list:
        """
        Parse all ids.
        :return: list with user ids, ex.: [123, 1234, ...]
        """
        users_dict = PARSER_ALL(
            self.METHOD,
            self.MAX_VALUE,
            values=self.params
        )
        ids = [user_dict['id'] for user_dict in users_dict['items']]
        return ids


if __name__ == "__main__":
    search_class = SearchIdsParser(q="Илья", online=1, has_photo=1, age_from=30, age_to=31,
                                   country="Россия", city="Москва")
    res = search_class.parse_all()
    print(len(res))
