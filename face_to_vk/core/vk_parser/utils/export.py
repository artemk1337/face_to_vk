import json


def str2txt(str_: str, filename: str = None):
    """

    :param str_:
    :type str_:
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    with open(filename, 'a+') as f:
        f.write("%s\n" % str_)


def list2txt(list_: list, filename: str = None):
    """

    :param list_:
    :type list_:
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    with open(filename, 'a+') as f:
        for item in list_:
            f.write("%s\n" % item)


def list2json(list_: list, filename: str = None):
    """
    Example: [{}, {}, ...]

    :param list_:
    :type list_:
    :param filename:
    :type filename:
    :return:
    :rtype:
    """
    with open(filename, 'w', encoding='UTF-8') as fp:
        json.dump(list_, fp, indent=4)
