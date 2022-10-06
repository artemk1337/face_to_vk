from typing import Optional

from db_connector.settings import STATUS_SET, STATUS_WAIT, STATUS_BUSY, STATUS_FINISH
from db_connector.singleton import CONN


class BaseConnector:
    TABLE_NAME = None
    SELECT_ROW_PARAMS = None

    @classmethod
    def _replace_busy_to_wait(cls):
        with CONN.cursor() as cur:
            # check status busy and switch to wait
            cur.execute(f"SELECT id FROM {cls.TABLE_NAME} WHERE status = '{STATUS_BUSY}'")
            rows = cur.fetchall()
            if rows:
                rows = tuple(row[0] for row in rows)
                cur.execute(f"UPDATE {cls.TABLE_NAME} SET status = '{STATUS_WAIT}' WHERE id in {rows}")
                CONN.commit()

    @classmethod
    def validate_on_start(cls):
        """
        Validate before first start
        """
        raise NotImplementedError

    @classmethod
    def _validate_status(cls, status: Optional[str]) -> None:
        """
        Validate status
        :param status: status
        """
        if status and status not in STATUS_SET:
            raise ValueError(f"Not valid status; available: {STATUS_SET}")

    @classmethod
    def delete_rows_by_status(cls, status: Optional[str] = STATUS_FINISH) -> None:
        cls._validate_status(status)
        with CONN.cursor() as cur:
            cur.execute(f"DELETE FROM {cls.TABLE_NAME} WHERE status = '{status}'")
            CONN.commit()

    @classmethod
    def delete_rows_by_ids(cls, ids: list) -> None:
        """
        Delete rows by ids
        :param ids:
        :return:
        """
        values = "(" + ", ".join(tuple(map(str, ids))) + ")"
        with CONN.cursor() as cur:
            cur.execute(f"DELETE FROM {cls.TABLE_NAME} WHERE id IN {values}")
            CONN.commit()

    @classmethod
    def delete_all_rows(cls) -> None:
        """
        Delete all rows
        """
        with CONN.cursor() as cur:
            cur.execute(f"DELETE FROM {cls.TABLE_NAME}")
            CONN.commit()

    @classmethod
    def update_status_by_id(cls, id_: int, status: str = STATUS_FINISH) -> None:
        """
        Update status row by id
        :param id_: id row
        :param status: new status
        :return:
        """
        cls._validate_status(status)
        with CONN.cursor() as cur:
            cur.execute(f"UPDATE {cls.TABLE_NAME} SET status = '{status}' WHERE id = '{id_}'")
            CONN.commit()

    @classmethod
    def select_wait_row(cls, switch_status: Optional[str] = STATUS_BUSY) -> Optional[tuple]:
        """
        Select one row with status 'wait'
        :param switch_status: change status of extracted row on new
        :return: SELECT_ROW_PARAMS for each class
        """
        cls._validate_status(switch_status)
        with CONN.cursor() as cur:
            # select only one row
            cur.execute(f"SELECT {cls.SELECT_ROW_PARAMS} FROM {cls.TABLE_NAME} WHERE status = '{STATUS_WAIT}' "
                        f"ORDER BY created_time LIMIT 1")
            rows = cur.fetchall()
            if not rows:
                return None
            row = rows[0]
            id_ = row[0]
            if switch_status:
                # update this one row
                cur.execute(f"UPDATE {cls.TABLE_NAME} SET status = '{switch_status}' WHERE id = '{id_}'")
                CONN.commit()
        return row

    @classmethod
    def insert(cls, *args) -> None:
        raise NotImplementedError
