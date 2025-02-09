# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# coding: utf8

import re

WITH_DEFINE_RE = re.compile(r'^.*(?:{.*})?.*$')
FLOAT_RE = re.compile(r'^-?\d*(?:{.*})?\d*(?:\.\d*(?:{.})?\d*)?$')
INT_RE = re.compile(r'^\d*(?:{.*})?\d*$')
REMOVE_BRACES_RE = re.compile(r'(.*){.*}(.*)')

def can_be_float(value, required=False, float_validator=None):
    """
    Check if given value can represent a valid float.
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or an empty string
    :param float_validator: optional validator of float value, callable which takes float and returns bool
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid float and float_validator is None or returns True for float represented by value
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value:
        return bool(FLOAT_RE.match(value))
    try:
        f = float(value)
        return True if float_validator is None else float_validator(f)
    except ValueError: return False


def can_be_int(value, required=False, int_validator=None):
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
    if '{' in value:
        return bool(INT_RE.match(value))
    try:
        i = int(value)
        return True if int_validator is None else int_validator(i)
    except ValueError: return False


def can_be_one_of(value, *values, **kwargs):
    """
    Check if given value can represent one of specified values
    :param str value: value to check, can be None
    :param bool required: if True, the function returns False if value is None or empty string
    :param bool case_sensitive: if True, the the comparison is case sensitive
    :param values: list of allowed values
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid boolean
    """
    required = kwargs.pop('required', False)
    case_sensitive = kwargs.pop('case_sensitive', False)
    if len(kwargs) > 0:
        raise TypeError("can_be_one_of() got an unexpected keyword argument '{}'".format(list(kwargs.keys())[0]))
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value:
        return bool(WITH_DEFINE_RE.match(value))
    if case_sensitive:
        return value in values
    else:
        return value.lower() in (val.lower() for val in values)


def can_be_bool(value, required=False):
    """
    Check if given value can represent a valid boolean ('true', 'false', 'yes', 'no', '1', '0').
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or empty string
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid boolean
    """
    return can_be_one_of(value, 'true', 'false', 'yes', 'no', '1', '0', required=required)


def can_be_double_float(value, required=False):
    """
    Check if given value can represent a valid double float (two floats separated by a space).
    :param str value: value to check, can be None
    :param bool required: only if True, the function returns False if value is None or empty string
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid double float
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value:
        return True
    parts = value.split()
    if len(parts) != 2: return False
    return FLOAT_RE.match(parts[0].strip()) and FLOAT_RE.match(parts[1].strip())


def can_be_list(value, separator=';', required=False, item_validator=None):
    """
    Check if given value can represent a valid list of items.
    :param str value: value to check, can be None
    :param str separator: separator of items in the list
    :param bool required: only if True, the function returns False if value is None or empty string
    :param item_validator: optional validator of list items, callable which takes item and returns bool
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid list of items and item_validator is None or returns True for all items in the list
    """
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value:
        return bool(WITH_DEFINE_RE.match(value))
    items = value.split(separator)
    if not items: return False
    return all(item_validator(item.strip()) if item_validator else True for item in items)
