import time

from parsers.vk_parser.parse import ParseMethods
from db_connector.commands.queue import QueueConnector
from db_connector.commands.users import UsersConnector


def queue_to_users() -> bool:
    """

    :return: status: True or False
    """
    status = False

    # get rows
    rows = QueueConnector.select_wait_row(limit=10)

    # clean queue
    if not rows:
        time.sleep(5)
        return status

    # process each row
    for row in rows:
        id_, user_id, my_uuid = row

        # parse if not exist in users
        if not UsersConnector.check_exist_user_id(user_id):

            # parse user
            user_info: dict = ParseMethods.parse_user_pages_with_images([user_id])[0]

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

    return status
