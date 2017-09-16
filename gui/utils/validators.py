# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation; either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import re

WITH_DEFINE_RE = re.compile(r'^.*(?:{.*})?.*$')
FLOAT_RE = re.compile(r'^\d*(?:{.*})?\d*(?:\.\d*(?:{.})?\d*)?$')
INT_RE = re.compile(r'^\d*(?:{.*})?\d*$')

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
    :param bool required: only if True, the function returns False if value is None or empty string
    :param values: list of allowed values
    :return bool: True only if:
        - value includes '{' or
        - value is None or empty and required is False or
        - value represents a valid boolean
    """
    required = kwargs.pop('required', False)
    if len(kwargs) > 0:
        raise TypeError("can_be_one_of() got an unexpected keyword argument '{}'".format(list(kwargs.keys())[0]))
    if value is None: return not required
    value = value.strip()
    if not value: return not required
    if '{' in value:
        return bool(WITH_DEFINE_RE.match(value))
    return value.lower() in values


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
