# -*- coding: utf-8 -*-
import re
import os
from subprocess import Popen, PIPE
import tempfile


TASKS_PATTERN = (r"#? ?TODO ?:?[^#]*|#? ?FIXME ?:?[^#]*|"
                 r"#? ?XXX ?:?[^#]*|#? ?HINT ?:?[^#]*|#? ?TIP ?:?[^#]*")


def tasks(project, source_code):
    """
    Find tasks in source code (TODO, FIXME, XXX, ...)

    :param project: project root folder
    :param source_code: string
    :return: list of strings
    """
    results = []
    for line, text in enumerate(source_code.splitlines()):
        for todo in re.findall(TASKS_PATTERN, text):
            results.append((todo, line + 1))
    return results


def flakes(project, source_code, filename=None):
    """
    Check source code with pyflakes

    :param project: project root folder
    :param source_code: string
    :param filename: relative of absolute filename (None if not saved)
    :return: list of strings (empty if pyflakes is not installed)

    """
    if filename is None:
        filename = '<string>'
        source_code += '\n'
    import _ast
    from pyflakes.checker import Checker
    # First, compile into an AST and handle syntax errors.
    try:
        tree = compile(source_code, filename, "exec", _ast.PyCF_ONLY_AST)
    except SyntaxError, value:
        # If there's an encoding problem with the file, the text is None.
        if value.text is None:
            return []
        else:
            return [(value.args[0], value.lineno)]
    else:
        # Okay, it's syntactically valid.  Now check it.
        w = Checker(tree, filename)
        w.messages.sort(lambda a, b: cmp(a.lineno, b.lineno))
        results = []
        lines = source_code.splitlines()
        for warning in w.messages:
            if 'analysis:ignore' not in lines[warning.lineno - 1]:
                results.append(
                    (warning.message % warning.message_args, warning.lineno))
        return results


def _check_external(args, source_code, filename=None, options=None):
    """
    Check source code with checker defined with *args* (list)
    Returns an empty list if checker is not installed
    """
    if args is None:
        return []
    if options is not None:
        args += options
    source_code += '\n'
    if filename is None:
        # Creating a temporary file because file does not exist yet
        # or is not up-to-date
        tempfd = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        tempfd.write(source_code)
        tempfd.close()
        args.append(tempfd.name)
    else:
        args.append(filename)
    output = Popen(args, stdout=PIPE, stderr=PIPE
                   ).communicate()[0].strip().splitlines()
    if filename is None:
        os.unlink(tempfd.name)
    return output


def pep8(project, source_code, filename=None):
    """
    Check source code with pep8

    :param project: project root folder
    :param source_code: string
    :param filename: relative of absolute filename (None if not saved)
    :return: list of strings (empty if pep8 is not installed)

    """
    # TODO: use internal pep8 instead?
    args = ['pep8']
    results = []
    source = source_code.splitline()
    for line in _check_external(args, source_code, filename=filename, options=['-r']):
        lineno = int(re.search(r'(\:[\d]+\:)', line).group()[1:-1])
        # nice idea from Spyder
        if 'analysis:ignore' not in source[lineno - 1]:
            message = line[line.find(': ') + 2:]
            results.append((message, lineno))
    return results


def lint(project, source_code, filename=None, errors_only=False):
    """
    Check source code with pylint

    :param project: project root folder
    :param source_code: string
    :param filename: relative of absolute filename (None if not saved)
    :param errors_only: boolean flag to disable style checks
    :return: list of strings (empty if pylint is not installed)
    """
    args = ["pylint"]
    options = ["-f parseable", "-r", "-n", "--disable-msg-cat=C,R"]
    if errors_only:
        options.append("-E")
    results = []
    for line in _check_external(args, source_code, filename=filename, options=options):
        match = re.search("\\[([WE])(, (.+?))?\\]", line)
        if match:
            kind = match.group(1)
            func = match.group(3)
            if kind == "W":
                msg = "Warning"
            else:
                msg = "Error"
            if func:
                line = re.sub("\\[([WE])(, (.+?))?\\]",
                              "%s (%s):" % (msg, func), msg)
            else:
                line = re.sub("\\[([WE])?\\]", "%s:" % msg, line)
            results.append((msg, line))
    return results
