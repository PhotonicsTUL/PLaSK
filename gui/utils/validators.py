
def can_be_float(value, required = False, float_validator = None):
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    try:
        f = float(value)
        return True if float_validator is None else float_validator(f)
    except ValueError: return False

def can_be_int(value, required = False, int_validator = None):
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    try:
        i = int(value)
        return True if int_validator is None else int_validator(i)
    except ValueError: return False

def can_be_bool(value, required = False):
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value: return True
    return value.lower() in ('true', 'false', 'yes', 'no', '1', '0')