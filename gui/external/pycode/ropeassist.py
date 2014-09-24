# -*- coding: utf-8 -*-
import sys
import os
import inspect
from zipimport import zipimporter

import libs  # use internal rope (might be newer)

import rope.base.libutils
import rope.base.project
import rope.contrib.codeassist


ROPE_DEFAULT_PREFS = {
    'automatic_soa': True,
    'automatic_soi': True,
    'ignore_syntax_errors': True,
    'ignore_bad_imports': True,
    'soa_followed_calls': 3,
    'perform_doa': True,
    'import_dynload_stdmods': True,
    'extension_modules': [
        "PyQt4", "PyQt4.QtGui", "QtGui", "PyQt4.QtCore", "QtCore",
        "PyQt4.QtScript", "QtScript", "os.path", "numpy", "scipy", "PIL",
        "OpenGL", "array", "audioop", "binascii", "cPickle", "cStringIO",
        "cmath", "collections", "datetime", "errno", "exceptions", "gc",
        "imageop", "imp", "itertools", "marshal", "math", "mmap", "msvcrt", "multiprocessing",
        "nt", "operator", "os", "parser", "rgbimg", "signal", "strop", "sys",
        "thread", "time", "wx", "wxPython", "xxsubtype", "zipimport", "zlib"
    ],
}


class Project(object):

    def __init__(self, root_path):
        self.root_path = root_path
        try:
            self._rope_prj = rope.base.project.Project(
                self.root_path, **ROPE_DEFAULT_PREFS)
        except Exception, _error:
            print "Could not create rope project", _error
            import traceback
            traceback.print_exc()
            #self._rope_prj = rope.base.project.get_no_project()
            self._rope_prj = rope.base.project.NoProject()
            self._rope_prj.root = None

    def validate(self):
        self._rope_prj.validate(self._rope_prj.root)

    def close(self):
        self._rope_prj.close()


_project_cache = {}


def _get_project(project):
    if isinstance(project, basestring):
        root_path = project
        try:
            project = _project_cache[root_path]
        except KeyError:
            project = _project_cache[root_path] = Project(root_path)
    return project


def _get_resource(project, filename):
    try:
        resource = rope.base.libutils.path_to_resource(
            project._rope_prj, filename.encode('utf-8'))
    except Exception, _error:
        resource = None
    return resource


def completions(project, source_code, offset, filename="XXXunknownXXX.py"):
    """
    Autocomplete

    :param project: project root folder
    :param source_code: source code string
    :param offset: absolute character offset
    :param filename: absolute or relative filename if exists
    :return: list of completition items
    """
    # TODO:
    #  * include import completitions
    #  * offer name to override from base after "def " inside a class
    #  *
    project = _get_project(project)
    resource = _get_resource(project, filename)
    try:
        proposals = rope.contrib.codeassist.code_assist(
            project._rope_prj, source_code, offset, resource)
        proposals = rope.contrib.codeassist.sorted_proposals(proposals)
        result = [(proposal.type, proposal.name) for proposal in proposals]
    except Exception, _error:
        print _error
        result = []
    return result


def calltip(project, source_code, offset, filename="XXXunknownXXX.py"):
    """
    Calltip (=signature of current call)

    :param project:
    :param source_code:
    :param offset:
    :param filename:
    :return: string, tuple
    """
    project = _get_project(project)
    resource = _get_resource(project, filename)
    try:
        cts = rope.contrib.codeassist.get_calltip(
            project._rope_prj, source_code, offset, resource, ignore_unknown=False, remove_self=True)
        if cts is not None:
            while '..' in cts:
                cts = cts.replace('..', '.')
            try:
                doc_text = rope.contrib.codeassist.get_doc(
                    project._rope_prj, source_code, offset, resource)
            except Exception, _error:
                print _error
                doc_text = ""
        else:
            return ("", "")
        return cts, doc_text
    except Exception, _error:
        print _error
    return ("", "")


def definition_location(project, source_code, offset, filename="XXXunknownXXX.py"):
    """
    Show where name if defined

    :param project: project root folder
    :param source_code:
    :param offset: absolute character offset
    :param filename:
    :return: (row, column) tuple
    """
    project = _get_project(project)
    resource = _get_resource(project, filename)
    try:
        resource, lineno = rope.contrib.codeassist.get_definition_location(
            project._rope_prj, source_code, offset, resource)
        if resource is not None:
            filename = resource.real_path
        return filename, lineno
    except Exception, _error:
        print _error
    return (None, None)


db = {}


def root_modules():
    """
    Returns a list containing the names of all the modules available in the
    folders of the pythonpath.
    """
    from time import time
    TIMEOUT_GIVEUP = 20
    modules = []
    if db.has_key('rootmodules'):
        return db['rootmodules']
    t = time()
    for path in sys.path:
        modules += module_list(path)
        if time() - t > TIMEOUT_GIVEUP:
            print "Module list generation is taking too long, we give up."
            print
            db['rootmodules'] = []
            return []

    modules += sys.builtin_module_names

    modules = list(set(modules))
    if '__init__' in modules:
        modules.remove('__init__')
        modules = list(set(modules))
        db['rootmodules'] = modules
    return modules


def module_list(path):
    """
    Return the list containing the names of the modules available in the given
    folder.
    """

    if os.path.isdir(path):
        folder_list = os.listdir(path)
    elif path.endswith('.egg'):
        try:
            folder_list = [f for f in zipimporter(path)._files]
        except:
            folder_list = []
    else:
        folder_list = []
        #folder_list = glob.glob(os.path.join(path,'*'))
        folder_list = [p for p in folder_list
                       if os.path.exists(os.path.join(path, p, '__init__.py'))
                       or p[-3:] in ('.py', '.so')
                       or p[-4:] in ('.pyc', '.pyo', '.pyd')]

    folder_list = [os.path.basename(p).split('.')[0] for p in folder_list]
    return folder_list


def module_completions(line):
    """
    Returns a list containing the completion possibilities for an import line.
    The line looks like this :
    'import xml.d'
    'from xml.dom import'
    """
    def tryImport(mod, only_modules=False):
        def isImportable(module, attr):
            if only_modules:
                return inspect.ismodule(getattr(module, attr))
            else:
                return not(attr[:2] == '__' and attr[-2:] == '__')
        try:
            m = __import__(mod)
        except:
            return []
        completion_list = []
        mods = mod.split('.')
        for module in mods[1:]:
            try:
                m = getattr(m, module)
            except:
                return []
        if (not hasattr(m, '__file__')) or (not only_modules) or\
           (hasattr(m, '__file__') and '__init__' in m.__file__):
            completion_list = [
                attr for attr in dir(m) if isImportable(m, attr)]
        completion_list.extend(getattr(m, '__all__', []))
        if hasattr(m, '__file__') and '__init__' in m.__file__:
            completion_list.extend(module_list(os.path.dirname(m.__file__)))
        completion_list = list(set(completion_list))
        if '__init__' in completion_list:
            completion_list.remove('__init__')
        return completion_list

    def dotCompletion(mod):
        if len(mod) < 2:
            return filter(lambda x: x.startswith(mod[0]), root_modules())

        completion_list = tryImport('.'.join(mod[:-1]), True)
        completion_list = filter(lambda x: x.startswith(mod[-1]),
                                 completion_list)
        completion_list = ['.'.join(mod[:-1] + [el]) for el in completion_list]
        return completion_list

    words = line.split(' ')

    if len(words) == 3 and words[0] == 'from':
        if words[2].startswith('i') or words[2] == '':
            return ['import ']
        else:
            return []

    if words[0] == 'import':
        if ',' == words[-1][-1]:
            return [' ']

        mod = words[-1].split('.')
        return dotCompletion(mod)

    if len(words) < 3 and (words[0] == 'from'):
        if len(words) == 1:
            return root_modules()

        mod = words[1].split('.')
        return dotCompletion(mod)

    if len(words) >= 3 and words[0] == 'from':
        mod = words[1]
        completion_list = tryImport(mod)
        if words[2] == 'import' and words[3] != '':
            if '(' in words[-1]:
                words = words[:-2] + words[-1].split('(')
            if ',' in words[-1]:
                words = words[:-2] + words[-1].split(',')
            return filter(lambda x: x.startswith(words[-1]), completion_list)
        else:
            return completion_list

    return []
