from typing import Optional

from core.db_connector.settings import CONN
from core.db_connector.commands.base import BaseConnector


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
    def check_exist_user_id(cls, user_id: int) -> bool:
        """
        Check exist user id in table

        :param user_id:
        :return:
        >>> UsersConnector.check_exist_user_id(123)
        """
        with CONN.cursor() as cur:
            cur.execute(f"SELECT id FROM {cls.TABLE_NAME} WHERE id = %s", (user_id,))
            return cur.rowcount > 0

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
