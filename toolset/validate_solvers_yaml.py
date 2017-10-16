#!/usr/bin/env python
import sys
import os
import jsonschema
import jsonschema.exceptions

from collections import OrderedDict

try:
    from ruamel import yaml

except ImportError:
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.iteritems())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    def text_constructor(loader, node):
        return loader.construct_scalar(node).strip().encode('utf-8')

    yaml.add_representer(OrderedDict, dict_representer)
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, dict_constructor)
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, text_constructor)

    import yaml
    def print_error(error, name, instance):
        error_format = "Error in {name} on {error_path}:\n\033[36m"\
                       "{bad_instance}\033[00m\n"\
                       "\033[31;01m{error.message}\033[00m\n\n"
        bad_instance = yaml.dump(error.instance, encoding='utf-8', allow_unicode=True,
                                 width=72, default_flow_style=False)
        sys.stderr.write(error_format.format(error=error, bad_instance=bad_instance,
                                             error_path=list(error.absolute_path), name=name))

    kwargs = {}

else:
    BEFORE = 3
    AFTER = 10

    def print_error(error, name, instance):
        instances = [instance]
        path = list(error.absolute_path)
        for i in path:
            instances.append(instances[-1][i])
        line = None
        for i in range(len(instances)-1, -1, -1):
            try:
                line = instances[i].lc.line
            except AttributeError:
                if i > 0:
                    try:
                        line = instances[i-1].lc.data[path[i-1]][0]
                    except (AttributeError, KeyError, IndexError):
                        pass
                    else:
                        break
            else:
                break
        if len(path) > 0:
            where = path[-1]
        else:
            where = ""
        if line is not None:
            lines = open(name).readlines()
            l0, l1 = max(line-BEFORE, 0), min(line+AFTER, len(lines))
            lines = lines[l0:l1]
            sys.stderr.write("Error in {} line {} [{}]:\n".format(name, line+1, where))
            line -= l0
            for i, text in enumerate(lines):
                sys.stderr.write("{}{:4d}:\033[36m{}\033[00m\n".format('\033[33;01m-> ' if i == line else '   ', l0+i+1,
                                                                    text.rstrip()))
        else:
            sys.stderr.write("Error in {} [{}]:\n\033[36m".format(name, where))
            sys.stderr.write(yaml.dump(error.instance, encoding='utf-8', allow_unicode=True, width=72, default_flow_style=False))
            sys.stderr.write("\033[00m\n")
        sys.stderr.write("ValidationError: \033[31;01m{}\033[00m\n\n".format(error.message))

    kwargs = dict(Loader=yaml.RoundTripLoader)

    import warnings
    warnings.simplefilter('ignore', yaml.error.UnsafeLoaderWarning)


schema_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'doc', 'schema', 'solvers.yml')
schema = yaml.load(open(schema_file))
validator = jsonschema.validators.validator_for(schema)(schema)


def validate(fname):
    instance = yaml.load(open(fname), **kwargs)
    error = jsonschema.exceptions.best_match(validator.iter_errors(instance))
    if error:
        print_error(error, fname, instance)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(validate(sys.argv[1]))
