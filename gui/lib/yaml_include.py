# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2023 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import os

import yaml


class AddYamlIncludePath:

    def __init__(self, path):
        self.path = path
        if yaml.__version__ < '5.0':
            for _loader in yaml.Loader, yaml.SafeLoader:
                _loader.add_constructor('!include', self.load_yaml_include)
        else:
            for _loader in yaml.Loader, yaml.FullLoader, yaml.SafeLoader, yaml.UnsafeLoader:
                _loader.add_constructor('!include', self.load_yaml_include)

    def load_yaml_include(self, loader, node):
        filename = None
        updates = None

        if isinstance(node, yaml.nodes.ScalarNode):
            filename = loader.construct_scalar(node)
        elif isinstance(node, yaml.nodes.MappingNode):
            kwargs = loader.construct_mapping(node, deep=True)
            filename = kwargs.get('$file', None)
            updates = kwargs.get('$update', None)
        else:
            raise TypeError('Un-supported YAML node {!r}'.format(node))

        if filename is None:
            raise TypeError('Missing filename in YAML node {!r}'.format(node))

        filename = os.path.normpath(filename)

        if not os.path.isabs(filename):
            filename = os.path.join(self.path, filename)

        with open(filename, 'r', encoding='utf8') as file:
            result = yaml.load(file, loader.__class__)

        if updates:
            for update in updates:
                path = update['$path']
                u = result
                for p in path[:-1]:
                    u = u[p]
                if isinstance(u, list) and path[-1] is None:
                    u.append(update['$value'])
                else:
                    u[path[-1]] = update['$value']

        return result
