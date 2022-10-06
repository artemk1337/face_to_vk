import time
from typing import Union, Optional

from parsers.vk_parser.user.format import UserInfoFormat
from parsers.vk_parser.base import BaseParser
from parsers.vk_parser.settings import VK_SESSION, PHOTO_MAX_SHIFT_TIME


class UsersPhotosParser(BaseParser):
    """
    Parse users photos with constraints.
    """

    METHOD = None

    def __init__(self, user_ids: list, max_shift_time=PHOTO_MAX_SHIFT_TIME):
        self.user_ids = user_ids
        self.max_shift_time = max_shift_time

    def _check_constraints(self, image: dict) -> bool:
        return time.time() - image['date'] < self.max_shift_time

    def parse_all(self) -> dict:
        """
        Parse all photos with constraints.
        :return: dict with ids and images, example: {user_id: [image1, image2, ...]}
        >>> UsersPhotosParser([1]).parse_all()
        """
        users_images = {}
        for user_id in self.user_ids:
            users_images[user_id]: list = VK_SESSION.get_api().photos.getAll(
                owner_id=user_id,
                extended=0
            )['items']
            users_images[user_id] = [image['sizes'][-1]['url'] for image in users_images[user_id]
                                     if self._check_constraints(image)]

        return users_images


class UsersMainInfoParser(BaseParser):
    """
    Parse main info from users page.
    """

    METHOD = None

    def __init__(
            self,
            user_ids: list,
            fields: Optional[Union[list, tuple, set]] = (
                    'about',
                    'bdate',
                    'has_photo',
                    'sex',
                    'last_seen',
                    'online',
                    'deactivated',
                    'can_access_closed',
                    'photo_400_orig',
            ),
    ):
        self.fields = fields or (
            'about',
            'bdate',
            'has_photo',
            'sex',
            'last_seen',
            'online',
            'deactivated',
            'can_access_closed',
            'photo_400_orig',
        )
        self.user_ids = user_ids

    def _check_constraints(self):
        pass

    def parse_all(self) -> dict:
        """
        Parse user main info.
        :return: dict, example: {id: {...}}
        >>> UsersMainInfoParser([1]).parse_all()
        """
        users = VK_SESSION.get_api().users.get(
            user_ids=self.user_ids,
            fields=self.fields,
            name_case='nom'
        )
        users_dict = {}
        for i, user in enumerate(users):
            user['images'] = [user['photo_400_orig']]
            users_dict[user["id"]] = UserInfoFormat(**user).dict()

        return users_dict
