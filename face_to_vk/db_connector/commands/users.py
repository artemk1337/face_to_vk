from typing import Optional, Union

from db_connector.singleton import CONN
from db_connector.commands.base import BaseConnector


class UsersConnector(BaseConnector):
    TABLE_NAME = "users"
    SELECT_ROW_PARAMS = "id, images"

    @classmethod
    def validate_on_start(cls):
        """
        Validate before first start
        """
        pass

    @classmethod
    def insert(
            cls,
            user_id: int,
            can_access_closed: bool,
            first_name: Optional[str] = None,
            last_name: Optional[str] = None,
            sex: Optional[int] = None,
            bdate: Optional[int] = None,
            images: Optional[list] = None,
    ):
        """
        Insert user info in table users
        :param user_id:
        :param can_access_closed:
        :param first_name:
        :param last_name:
        :param sex:
        :param bdate:
        :param images:
        :return:
        >>> UsersConnector.insert(123, True)
        """
        kwargs = {col: str(val) for col, val in zip(
                ("id", "can_access_closed", "first_name", "last_name", "sex", "bdate", "images"),
                (user_id, can_access_closed, first_name, last_name, sex, bdate, images)) if val}
        cols = '(' + ', '.join(kwargs.keys()) + ')'
        vals = '(' + ', '.join(kwargs.values()) + ')'
        with CONN.cursor() as cur:
            cur.execute(f"INSERT INTO {cls.TABLE_NAME} {cols} VALUES {vals}")
            CONN.commit()
