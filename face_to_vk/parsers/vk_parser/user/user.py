import time
from typing import Union, Optional

from pydantic import BaseModel as BaseFormat, validator

from settings import VK_SESSION, TOOLS, ITER_MAX_BUFFER, PHOTO_MAX_SHIFT_TIME, PARSER_ALL


class UserInfoFormat(BaseFormat):
    """
    Validate format user page info.
    """

    id: int
    can_access_closed: bool
    sex: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    bdate: Optional[str] = None
    last_seen: Optional[Union[int, dict]] = None
    images: Optional[list] = None

    @validator("last_seen")
    def last_seen_reformat(cls, value):
        if value:
            return value.get('time', None)
        return value


class UserPhotosParser:
    """
    Parse all user photos.
    """

    def __init__(self, user_id, max_shift_time=PHOTO_MAX_SHIFT_TIME):
        self.user_id = user_id
        self.max_shift_time = max_shift_time

    def parse_all(self) -> list:
        images = VK_SESSION.get_api().photos.getAll(
            owner_id=self.user_id,
            extended=0
        )['items']
        images = [image['sizes'][-1]['url'] for image in images if time.time() - image['date'] < self.max_shift_time]

        return images


class UserMainInfoParser:
    """
    Parse user page main info.
    """

    def __init__(
            self,
            user_ids: list,
            fields: Union[list, tuple, set] = (
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
        self.user_ids = user_ids
        self.fields = fields

    def parse_all(self):
        users = VK_SESSION.get_api().users.get(
            user_ids=self.user_ids,
            fields=self.fields,
            name_case='nom'
        )
        for i, user in enumerate(users):
            user['images'] = [user['photo_400_orig']]
            users[i] = UserInfoFormat(**user).dict()

        return users


class UserFriendsIdsParser:
    """
    Parse all friend ids.
    """

    METHOD_FRIENDS = 'friends.get'

    def __init__(self, user_id: int):
        self.user_id = user_id

    def parse_all(self) -> list:
        users = PARSER_ALL(
            method=self.METHOD_FRIENDS,
            max_count=ITER_MAX_BUFFER,
            values={'user_id': self.user_id}
        )
        return users['items']
