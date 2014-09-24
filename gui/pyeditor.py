from .external.pycode import pyqtfrontend


class PyEdit(pyqtfrontend.PyCode):

    def __init__(self, project_folder, textedit, filename=None, prefix=""):
        super(PyEdit, self).__init__(project_folder, textedit, filename)
        self.set_prefix(prefix)

    def source(self):
        src, pos = super(PyEdit, self).source()
        src = self.prefix + src
        pos = pos + len(self.prefix)
        return src, pos

    def set_prefix(self, prefix):
        self.prefix = prefix
        if len(self.prefix) > 0 and self.prefix[-1] != '\n':
            self.prefix += '\n'
