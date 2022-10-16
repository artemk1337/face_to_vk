from typing import Union, Optional

from core.vk_parser.user import UsersPhotosParser, UsersMainInfoParser
from core.vk_parser.group import GroupIdsParser
from core.vk_parser.search import SearchIdsParser


class ParseMethods:
    @staticmethod
    def parse_user_pages_with_images(ids: list, fields: Optional[Union[list, tuple, set]] = None) -> list:
        """
        Parse user info and images
        :return: return list with dicts
        >>> ParseMethods.parse_user_pages_with_images([1])
        """
        users = UsersMainInfoParser(ids, fields).parse_all()
        photos = UsersPhotosParser(ids).parse_all()
        for user_id, user in users.items():
            user['images'] += photos[user_id]
        return users

    @staticmethod
    def parse_user_ids_from_group(group_id: int) -> list:
        """
        Parse user ids from group
        :param group_id: group id
        :return: user ids
        """
        return GroupIdsParser(group_id).parse_all()

    @staticmethod
    def parse_ids_from_search(q: str, **kwargs) -> list:
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
        :return: user ids
        """
        return SearchIdsParser(q=q, **kwargs).parse_all()
