from typing import Union, Optional

from user import UsersPhotosParser, UsersMainInfoParser
from group import GroupIdsParser
from search import SearchIdsParser

from utils.country_city_transformer import CountryTransformer, CityTransformer


def parse_user_pages_with_images(ids: list, fields: Optional[Union[list, tuple, set]] = None) -> list:
    """

    :return: return list with dicts
    >>> parse_user_pages_with_images([1])
    """
    users = UsersMainInfoParser(ids, fields).parse_all()
    photos = UsersPhotosParser(ids).parse_all()
    for user_id, user in users.items():
        user['images'] += photos[user_id]
    return users


def parse_user_ids_from_group(group_id):
    pass


def parse_ids_from_search(q: str, **kwargs) -> list:

    search_class = SearchIdsParser(q=q, **kwargs)
    ids = search_class.parse_all()
    return ids


