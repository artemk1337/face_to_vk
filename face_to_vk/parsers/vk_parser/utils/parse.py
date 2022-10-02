from typing import Union

from user.user import UserPhotosParser, UserMainInfoParser


def parse_profiles_images_by_id(ids: list) -> list:
    for id_dict in ids:
        id_dict['images'] += UserPhotosParser(id_dict['id']).parse_all()
    return ids


def parse_profiles(ids: Union[list, tuple, set]) -> list:
    return UserMainInfoParser(ids).parse_all()


def parse_profiles_with_images(ids: Union[list, tuple, set]) -> list:
    profiles = parse_profiles(ids)
    result = parse_profiles_images_by_id(profiles)
    return result
