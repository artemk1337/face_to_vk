from typing import Optional

from ..db_settings import DB_LOGGER as LOGGER, STATUS_SET, STATUS_WAIT, STATUS_BUSY, STATUS_FINISH, CONN


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
    def select_wait_row(cls, limit: int = 1, switch_status: Optional[str] = STATUS_BUSY) -> Optional[tuple]:
        """
        Select one row with status 'wait'

        :param limit: how many select
        :param switch_status: change status of extracted row on new
        :return: list with SELECT_ROW_PARAMS for each class
        """
        cls._validate_status(switch_status)
        with CONN.cursor() as cur:
            # select only one row
            cur.execute("BEGIN")
            cur.execute(f"SELECT {cls.SELECT_ROW_PARAMS} FROM {cls.TABLE_NAME} WHERE status = '{STATUS_WAIT}' "
                        f"ORDER BY created_time LIMIT {limit}")
            rows = cur.fetchall()
            LOGGER.info(str(rows))
            if not rows:
                cur.execute("COMMIT")
                return None
            ids = "(" + ', '.join([str(row[0]) for row in rows]) + ")"
            if switch_status:
                # update this one row
                cur.execute(f"UPDATE {cls.TABLE_NAME} SET status = '{switch_status}' WHERE id IN {ids}")
                CONN.commit()
            cur.execute("COMMIT")
        return rows

    @classmethod
    def insert(cls, *args) -> None:
        raise NotImplementedError
