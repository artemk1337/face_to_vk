from typing import Optional

from core.db_connector.settings import CONN


class VectorsConnector:
    TABLE_NAME = "vectors"

    @classmethod
    def delete_by_user_id(cls, user_id: int) -> None:
        """

        :param user_id:
        :return:
        >>> VectorsConnector.delete_by_user_id(123)
        """
        with CONN.cursor() as cur:
            cur.execute(f"DELETE FROM {cls.TABLE_NAME} WHERE user_id = %s", (user_id,))
            CONN.commit()

    @classmethod
    def insert(cls, user_id: int, vector: list, ):
        """

        :param user_id:
        :param vector:
        :return:
        >>> VectorsConnector.insert(123, [2., 3.3])
        """
        with CONN.cursor() as cur:
            cur.execute(f"INSERT INTO {cls.TABLE_NAME} (user_id, vector) VALUES (%s, %s)", (user_id, vector))
            CONN.commit()

    @classmethod
    def select(cls, size: int = 1000, scroll: int = 0) -> Optional[list]:
        """
        Select vectors with ids
        :param size: size for cur.fetchmany https://www.psycopg.org/docs/cursor.html#cursor.fetchmany
        :param scroll: scroll before fetch https://www.psycopg.org/docs/cursor.html#cursor.scroll
        :return: list with (id, user_id, vector)
        >>> VectorsConnector.select()
        """
        with CONN.cursor() as cur:
            cur.execute(f"SELECT id, user_id, vector FROM {cls.TABLE_NAME}")
            if cur.rowcount <= scroll:
                return None
            cur.scroll(scroll)
            return cur.fetchmany(size)


class BestVectorConnector:
    TABLE_NAME = "best_vectors"

    @classmethod
    def delete_by_user_id(cls, user_id: int) -> None:
        """

        :param user_id:
        :return:
        >>> BestVectorConnector.delete_by_user_id(123)
        """
        with CONN.cursor() as cur:
            cur.execute(f"DELETE FROM {cls.TABLE_NAME} WHERE user_id = %s", (user_id,))
            CONN.commit()

    @classmethod
    def insert(cls, user_id: int, vector_id: int):
        """

        :param user_id:
        :param vector_id:
        :return:
        >>> VectorsConnector.insert(123, 1)
        """
        with CONN.cursor() as cur:
            cur.execute(f"INSERT INTO {cls.TABLE_NAME} (user_id, vector_id) VALUES (%s, %s)", (user_id, vector_id))
            CONN.commit()

    @classmethod
    def select(cls, size: int = 1000, scroll: int = 0) -> Optional[list]:
        """
        Select vectors with ids
        :param size: size for cur.fetchmany https://www.psycopg.org/docs/cursor.html#cursor.fetchmany
        :param scroll: scroll before fetch https://www.psycopg.org/docs/cursor.html#cursor.scroll
        :return: list with (user_id, vector)
        >>> VectorsConnector.select()
        """
        with CONN.cursor() as cur:
            cur.execute(f"SELECT user_id, vector_id FROM {cls.TABLE_NAME}")
            if cur.rowcount <= scroll:
                return None
            cur.scroll(scroll)
            return cur.fetchmany(size)

