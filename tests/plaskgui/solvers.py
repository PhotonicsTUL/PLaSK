import plask


class Generic2D(plask.Solver):
    def load_xpl(self, xpl, manager):
        for tag in xpl:
            plask.print_log('detail', tag.name, tag.attrs)


class Configured2D(plask.Solver):
    def load_xpl(self, xpl, manager):
        for tag in xpl:
            name = tag.name
            plask.print_log('detail', name, tag.attrs)
            for tag in tag:
                plask.print_log('detail', name, '>', tag.name, tag.attrs)
