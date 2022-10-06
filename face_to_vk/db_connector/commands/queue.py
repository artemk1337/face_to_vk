from db_connector.settings import CONN
from db_connector.commands.base import BaseConnector


class QueueConnector(BaseConnector):
    TABLE_NAME = "queue"
    SELECT_ROW_PARAMS = "id, user_id, uuid"

    @classmethod
    def validate_on_start(cls):
        """
        Validate before first start
        """
        cls.delete_rows_by_status()

    @classmethod
    def insert(cls, user_ids: list, uuid: str) -> None:
        """
        Insert user ids in table queue

        :param user_ids:
        :param uuid:
        :return:
        >>> QueueConnector.insert([1, 2], "fds42")
        """
        values = ", ".join([f"('{user_id}', '{uuid}')" for user_id in user_ids])
        with CONN.cursor() as cur:
            cur.execute(f"INSERT INTO {cls.TABLE_NAME} (user_id, uuid) VALUES {values}")
            CONN.commit()


# few tests
if __name__ == "__main__":
    QueueConnector.insert([1, 2, 3], "c3901ucaasiu4398c")
    res1 = QueueConnector.select_wait_row()
    res2 = QueueConnector.select_wait_row()
    res3 = QueueConnector.select_wait_row()
    assert (res1[1] == 1 and res2[1] == 2 and res3[1] == 3)
    QueueConnector.delete_all_rows()
    res = QueueConnector.select_wait_row()
    assert res is None
