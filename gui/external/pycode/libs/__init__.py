# -*- coding: utf-8 -*-
"""
For ease of use this package contains copies of required libraries
"""
import sys
import os

path = os.path.dirname(os.path.abspath(__file__))

for name in os.listdir(path):
    if name.endswith(".zip"):
        zip_module = os.path.join(path, name)
        sys.path.insert(0, zip_module)
