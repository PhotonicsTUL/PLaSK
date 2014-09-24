# -*- coding: utf-8 -*-
import sys
import os
import fnmatch


class ProjectImporter(object):

    def guess_source_folders(self, root):
        src_folder = os.path.join(root, "src")
        if os.path.isdir(src_folder):
            return [src_folder]
        else:
            # XXX: use heuristic to find packages?
            return [root]

    def settings(self, filename):
        prefs = {}
        prefs["project_name"] = self.project_name(filename)
        prefs["project_root"] = self.project_root(filename)
        parsed = self.load(filename)
        prefs.update(self.evaluate(parsed))
        prefs["source_folders"] = prefs.get(
            "source_folders", None) or self.guess_source_folders(prefs["project_root"])
        prefs["interpreter"] = Interpreter()
        return prefs


class ProjectDirectoryImporter(ProjectImporter):

    directory = None

    def project_root(self, path):
        return os.path.dirname(path)

    def project_name(self, path):
        return os.path.basename(os.path.dirname(path))

    def discover(self, path):
        filename = os.path.join(path, self.directory)
        if os.path.exists(filename):
            if self.validate(filename):
                return [filename]

    def validate(self, path):
        return os.path.isdir(path)


class ProjectDirectoryFileImporter(ProjectDirectoryImporter):

    filename = None

    def project_root(self, filename):
        return os.path.dirname(filename)

    def project_name(self, path):
        return os.path.basename(os.path.dirname(path))

    def discover(self, path):
        filename = os.path.join(path, self.filename)
        if os.path.exists(filename):
            if self.validate(filename):
                return [filename]

    def validate(self, path):
        return not os.path.isdir(path)


def list_dir(path):
    # TODO: add caching
    try:
        return os.listdir(path)
    except OSError:
        return []


class ProjectFileImporter(ProjectImporter):

    match = None

    def discover(self, path):
        found = []
        for name in list_dir(path):
            if fnmatch.fnmatch(name, self.match):
                found.append(
                    (os.path.splitext(name)[0], os.path.join(path, name)))
        return found

    def validate(self, path):
        return not os.path.isdir(path)

    def project_root(self, filename):
        return os.path.dirname(filename)

    def project_name(self, filename):
        return os.path.splitext(os.path.basename(filename))[0]


class RopeProjectImporter(ProjectDirectoryImporter):

    directory = ".ropeproject"

    def load(self, path):
        class RopePreferences(dict):

            def add(self, key, value):
                self.setdefault(key, []).append(value)
        settings_py = os.path.join(path, "settings.py")
        ns = {}
        execfile(settings_py, ns)
        prefs = RopePreferences()
        ns["set_prefs"](prefs)
        return prefs

    def evaluate(self, prefs):
        return {
            "python_path": prefs.get("python_path", []),
            "source_folders": prefs.get("source_folders", []),

        }


class NinjaIdeProjectImporter(ProjectFileImporter):

    match = "*.nja"

    def validate(self, filename):
        return not os.path.isdir(filename)

    def load(self, filename):
        import json
        prefs = json.load(open(filename))
        return prefs

    def evaluate(self, prefs):
        return {
            "name": prefs.get("name", None),
            "description": prefs.get("description", None),
            "python_path": prefs.get("PYTHONPATH", "").split(":"),
            "executable": prefs.get("pythonPath", None),
            "virtualenv": prefs.get("venv", None),
        }


def etree(filename):
    from xml.etree.ElementTree import ElementTree
    xml = open(filename)
    return ElementTree.parse(xml)


class PyCharmProjectImporter(ProjectDirectoryImporter):

    directory = ".idea"

    def load(self, filename):
        #xml = etree(filename)
        return {}

    def evaluate(self, prefs):
        return {}


class PyDevProjectImporter(ProjectDirectoryFileImporter):

    filename = ".pydevproject"

    def load(self, filename):
        xml = etree(filename)
        root = xml.find("pydev_project")
        for prop in root.findall("pydev_property"):
            key = prop["name"]
        return {}

    def evaluate(self, prefs):
        return {}


class SpyderProjectImporter(ProjectDirectoryFileImporter):
    filename = ".spyderproject"

    def load(self, filename):
        import pickle
        prefs = pickle.load(filename)
        return prefs

    def evaluate(self, prefs):
        return {}


class PyCodeProjectImporter(ProjectDirectoryImporter):

    directory = ".pycode"


class AnyProjectImporter(ProjectImporter):
    classes = [PyCodeProjectImporter,
               PyCharmProjectImporter, NinjaIdeProjectImporter, RopeProjectImporter, SpyderProjectImporter]

    def discover(self, path):
        self.prj = None
        projects = [cls() for cls in self.classes]
        for prj in projects:
            found = prj.discover(path)
            if found:
                self.prj = prj
                return found

    def project_name(self, filename):
        return self.prj.project_name(filename)

    def project_root(self, filename):
        return self.prj.project_root(filename)

    def load(self, filename):
        return self.prj.load(filename)

    def evaluate(self, prefs):
        return self.prj.evaluate(prefs)


class Interpreter(object):

    def __init__(self):
        self.executeable = sys.executable
        self.python_path = sys.path
        self.version = sys.version_info

    def __repr__(self):
        version = ".".join(map(str, tuple(self.version)))
        return "<%s %s (%s)>" % (self.__class__.__name__, self.executeable, version)


class VirtualEnvInterpreter(object):

    def __init__(self, venv_bin_python):
        self.executable = venv_bin_python
        self.python_path = self._discover_path()

    def _pipe(self, cmd):
        output = os.popen('%s -c "%s"' % (self.executable, cmd)).read()
        return output.strip()

    def _discover_version(self):
        out = self._pipe(
            'import sys; print(tuple(sys.version_info))"' % self.executable)
        try:
            return sys.version_info(*eval(out))
        except:
            pass

    def _discover_paths(self):
        out = self._pipe("print(sys.path)")
        try:
            path_list = eval(out)
            assert isinstance(path_list, list)
            return path_list
        except:
            pass


class ProjectSettings(object):

    def __init__(self):
        self.interpreter = Interpreter()
        self.source_folders = []
        self.extra_path = []


def discover_project(filename, project_importer_class=PyCodeProjectImporter):
    # TODO: add caching
    prj = project_importer_class()
    path = os.path.abspath(filename)
    while path:
        path = os.path.dirname(path)
        found = prj.discover(path)
        if not found:
            continue
        for settings_filename in found:
            print "found", settings_filename
            return prj.settings(settings_filename)


if __name__ == "__main__":
    print discover_project(__file__, AnyProjectImporter)
