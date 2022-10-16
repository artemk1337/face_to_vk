from typing import Optional

from pydantic import BaseModel as BaseFormat, validator, root_validator

from core.vk_parser.utils.country_city_transformer import CountryTransformer, CityTransformer


class SearchFormat(BaseFormat):
    """
    Validate format user page info.
    https://dev.vk.com/method/users.search
    """

    q: str
    sort: int = 0
    sex: Optional[int]
    country: Optional[int]
    city: Optional[int]
    home_town: Optional[int]
    university_country: Optional[int]
    age_from: Optional[int]
    age_to: Optional[int]
    online: Optional[int]
    has_photo: Optional[int]
    from_list: Optional[str]

    @validator("sort")
    def sort_check(cls, value):
        if value in (0, 1):
            return value
        raise ValueError("Value must be 0 or 1")

    @validator("sex")
    def sex_check(cls, value):
        if value in (0, 1, 2):
            return value
        raise ValueError("Value must be 0, 1 or 2")

    @validator("online")
    def online_check(cls, value):
        if value in (0, 1):
            return value
        raise ValueError("Value must be 0 or 1")

    @validator("has_photo")
    def has_photo_check(cls, value):
        if value in (0, 1):
            return value
        raise ValueError("Value must be 0 or 1")

    @validator("from_list")
    def from_list_check(cls, value):
        if value in ('friends', 'subscriptions'):
            return value
        raise ValueError("Value must be 'friends' or 'subscriptions'")

    @validator("country", pre=True)
    def country_pre_check(cls, value):
        if isinstance(value, str):
            return CountryTransformer().name2id(value)  # set id country
        return value

    @root_validator(pre=True)
    def country_city_pre_check(cls, values):
        if values.get('country') and isinstance(values.get('country'), str):
            values['country'] = CountryTransformer().name2id(values.get('country'))  # set id country
            if values.get('city') and isinstance(values.get('city'), str):
                values['city'] = CityTransformer(values['country'], q=values.get('city')).name2id()  # set id city
            if values.get('home_town') and isinstance(values.get('home_town'), str):
                values['home_town'] = CityTransformer(values['country'], q=values.get('home_town')).name2id()  # set id city
        return values
