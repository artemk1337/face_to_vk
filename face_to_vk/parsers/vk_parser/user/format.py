from typing import Optional, Union

from pydantic import BaseModel as BaseFormat, validator


class UserInfoFormat(BaseFormat):
    """
    Validate format user page info.
    """

    id: int
    can_access_closed: bool
    sex: int = 0
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
