import time

from settings import MAIN_LOGGER as LOGGER
from core.vk_parser import VKParseMethods
from core.db_connector.commands.queue import QueueConnector
from core.db_connector.commands.users import UsersConnector


def queue_to_users() -> bool:
    """

    :return: status: True or False
    """
    status = False

    LOGGER.info("Started check")
    # get rows
    rows = QueueConnector.select_wait_row(limit=10)

    # clean queue
    if not rows:
        seconds_sleep = 10
        LOGGER.info(f"Sleep {seconds_sleep} seconds...")
        time.sleep(seconds_sleep)
        return status

    LOGGER.info(f"Processing rows...")
    # process each row
    for i, row in enumerate(rows):
        LOGGER.info(f"Processing row {i}...")
        id_, user_id, my_uuid = row

        # parse if not exist in users
        if not UsersConnector.check_exist_user_id(user_id):

            # parse user
            user_info: dict = VKParseMethods.parse_user_pages_with_images([user_id])[0]

            # insert in users
            UsersConnector.insert(
                user_id=user_id,
                can_access_closed=user_info.get('can_access_closed', None),
                first_name=user_info.get('first_name', None),
                last_name=user_info.get('last_name', None),
                sex=user_info.get('sex', None),
                bdate=user_info.get('bdate', None),
                images=user_info.get('images', None)
            )

            status = True

        # update status in queue
        QueueConnector.update_status_by_id(id_)
    LOGGER.info(f"Precessed rows")

    return status
