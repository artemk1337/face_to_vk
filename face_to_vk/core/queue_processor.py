import os

from settings import BASEDIR
from core.utils.queue_to_users import queue_to_users


def process():
    while True:
        queue_to_users()


if __name__ == "__main__":
    with open(os.path.join(BASEDIR, "core/queue.pid"), mode="w") as f:
        f.write(str(os.getpid()))
    process()
