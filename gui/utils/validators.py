
def can_be_float(value, required=False, float_validator=None):
    """
    Check if given value can represent a valid float.
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or empty string
    :param float_validator: optional validator of float value, callable which takes float and returns bool
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid float and float_validator is None or returns True for float represented by value
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    try:
        f = float(value)
        return True if float_validator is None else float_validator(f)
    except ValueError: return False

def can_be_int(value, required = False, int_validator = None):
    """
    Check if given value can represent a valid integer.
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or empty string
    :param int_validator: optional validator of integer value, callable which takes int and returns bool
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid integer and int_validator is None or returns True for integer represented by value
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    try:
        i = int(value)
        return True if int_validator is None else int_validator(i)
    except ValueError: return False

def can_be_bool(value, required = False):
    """
    Check if given value can represent a valid boolean ('true', 'false', 'yes', 'no', '1', '0').
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or empty string
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid boolean
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    return value.lower() in ('true', 'false', 'yes', 'no', '1', '0')
