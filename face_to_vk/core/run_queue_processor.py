from settings import MAIN_LOGGER as LOGGER
from core.db_transfer.queue_to_users import queue_to_users


def process():
    """
    Start process
    :return:
    """
    LOGGER.info("Started queue_processor")
    while True:
        queue_to_users()


if __name__ == "__main__":
    process()
