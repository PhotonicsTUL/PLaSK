def empty_to_none(str):
    """ :param str str: sring
        :return: None if str is empty or consists only with white characters, str in other cases"""
    return None if len(str) == 0 or str.isspace() else str