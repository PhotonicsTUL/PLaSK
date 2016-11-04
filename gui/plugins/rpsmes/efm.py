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

try:
    import gui
    from gui.qt.QtCore import Qt
    from gui.qt.QtWidgets import *
    from gui.qt.QtGui import *
except ImportError:
    qt = False
else:
    qt = True

import sys
import os

try:
    next = next
except NameError:
    _raiseStopIteration = object()
    def next(iterator, default=_raiseStopIteration):
        if not hasattr(iterator, 'next'):
            raise TypeError("not an iterator")
        try:
            return iterator.next()
        except StopIteration:
            if default is _raiseStopIteration:
                raise
            else:
                return default


class Material(object):

    def __new__(cls, materials, name, nr, ar, ng, ag, tnr, tar, lam0):
        obj = object.__new__(cls)
        obj.name = name
        obj.nr = nr
        obj.dn = (nr - ng) / lam0
        obj.tnr = tnr
        obj.ar = - ar
        obj.da = - (ar - ag) / lam0
        obj.tar = - tar
        idx = 1
        while obj.name in materials:
            if materials[obj.name] == obj:
                return materials[obj.name]
            else:
                obj.name = '{}_{:02d}'.format(name, idx)
                idx += 1
        materials[obj.name] = obj
        return obj

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Layer(object):
    def __init__(self, d, *mats):
        self.d = d * 1e4
        self.mats = mats


def reduce_layers(layers):
    if len(layers) < 2: return layers
    result = []
    cur = layers[0]
    for nxt in layers[1:]:
        for c,n in zip(cur.mats, nxt.mats):
            if c != n:
                result.append(cur)
                cur, nxt = nxt, None
                break
        if nxt is not None:
            cur.d += nxt.d
    result.append(cur)
    return result


def write_xpl(fname, materials, layers, lam0, lpm):

    ofile = open(fname, 'w')

    def out(text=''):
        ofile.write(text)
        ofile.write('\n')

    out('<plask>\n')

    out('<defines>')
    out('  <define name="dr1" value="1#changeme"/>')
    out('  <define name="dr2" value="1#changeme"/>')
    out('  <define name="dr3" value="1."/>')
    out('</defines>')

    materials = materials.values()
    materials.sort(key=lambda m: m.name)

    out('<materials>')
    for mat in materials:
        out('  <material name="{}" base="semiconductor">'.format(mat.name))
        if mat.dn == 0: dn = ''
        else: dn = '{m.dn:+g}*(wl-{l})'.format(m=mat, l=lam0)
        if mat.tnr == 0: tn = ''
        else: tn = '{m.tnr:+g}*(T-300.)'.format(m=mat)
        out('    <nr>{m.nr}{dn}{tn}</nr>'.format(m=mat, dn=dn, tn=tn))
        if mat.da == 0: da = ''
        else: da = '{m.da:+g}*(wl-{l})'.format(m=mat, l=lam0)
        if mat.tar == 0: ta = ''
        else: ta = '{m.tar:+g}*(T-300.)'.format(m=mat)
        out('    <absp>{m.ar}{da}{ta}</absp>'.format(m=mat, da=da, ta=ta))
        out('  </material>')
    out('</materials>\n')

    out('<geometry>')
    out('  <cylindrical axes="r,z" name="optical" outer="extend">')
    out('    <stack>')
    for layer in layers:
        if layer.mats[0] == layer.mats[1] == layer.mats[2]:
            out('      <rectangle dr="{{dr1+dr2+dr3}}" dz="{0.d}" material="{0.mats[0].name}"/>'.format(layer))
        else:
            out('      <shelf>')
            if layer.mats[1] == layer.mats[2]:
                out('        <rectangle dr="{{dr1}}" dz="{0.d}" material="{0.mats[0].name}"/>'.format(layer))
                out('        <rectangle dr="{{dr2+dr3}}" dz="{0.d}" material="{0.mats[1].name}"/>'.format(layer))
            elif layer.mats[0] == layer.mats[1]:
                out('        <rectangle dr="{{dr1+dr2}}" dz="{0.d}" material="{0.mats[0].name}"/>'.format(layer))
                out('        <rectangle dr="{{dr3}}" dz="{0.d}" material="{0.mats[2].name}"/>'.format(layer))
            else:
                out('        <rectangle dr="{{dr1}}" dz="{0.d}" material="{0.mats[0].name}"/>'.format(layer))
                out('        <rectangle dr="{{dr2}}" dz="{0.d}" material="{0.mats[1].name}"/>'.format(layer))
                out('        <rectangle dr="{{dr3}}" dz="{0.d}" material="{0.mats[2].name}"/>'.format(layer))
            out('      </shelf>')
    out('    </stack>')
    out('  </cylindrical>')
    out('</geometry>\n')

    out('<solvers>')
    out('<optical name="EFM" solver="EffectiveFrequencyCyl">')
    out('  <geometry ref="optical"/>')
    out('  <mode lam0="{}"/>'.format(lam0))
    out('</optical>')
    out('</solvers>\n')

    out('<script><![CDATA[')
    out('mn = EFM.find_mode({:.1f}, m={})\n'.format(lam0, lpm))
    out('print_log(\'result\', "Found mode at lam = {:.3f}nm, loss = {:.2f}/cm"')
    out('    .format(EFM.outWavelength(mn), EFM.outLoss(mn)))\n')
    out('msh = mesh.Rectangular2D(')
    out('    mesh.Regular(0., dr1+dr2+dr3, 501),')
    out('    mesh.Regular(GEO.optical.bbox.lower.z, GEO.optical.bbox.upper.z, 501))\n')
    out('plot_field(EFM.outLightMagnitude(msh))')
    out('plot_geometry(GEO.optical, color=\'0.5\', alpha=0.35)\n')
    out('show()')
    out(']]></script>\n')
    out('</plask>')


def read_efm(fname):

    ifile = open(fname)

    # Set-up generator, which skips empty lines, strips the '\n' character, and splits line by tabs
    def Input(ifile):
        for line in ifile:
            if line[-1] == "\n": line = line[:-1]
            if line.strip(): yield line.split()
    input = Input(ifile)

    def skip(n):
        for _ in range(n):
            next(input)

    # Read header
    skip(3)
    _, nl = next(input); nl = int(nl)
    skip(5)
    _, lam0 = next(input); lam0 = 1e3 * float(lam0)
    skip(5)
    _, m = next(input)
    skip(5)

    materials = {}
    layers = []

    for i in range(nl):
        name0, name2, name1 = next(input)
        nr0, ar0, ng0, ag0, nr2, ar2, ng2, ag2, d = map(float, next(input)[2::2])
        tnr0, tar0, _, _, tnr2, tar2, _, _ = map(float, next(input)[1::2])
        nr1, ar1, ng1, ag1, tnr1, tar1, _, _ = map(float, next(input)[1::2])
        mat0 = Material(materials, name0, nr0, ar0, ng0, ag0, tnr0, tar0, lam0)
        mat1 = Material(materials, name1, nr1, ar1, ng1, ag1, tnr1, tar1, lam0)
        mat2 = Material(materials, name2, nr2, ar2, ng2, ag2, tnr2, tar2, lam0)
        layers.append(Layer(d, mat0, mat1, mat2))

    layers = reduce_layers(layers)

    return materials, layers, lam0, m


if qt:

    def import_efm(parent):
        """Convert .efm file to .xpl, save it to disk and open in PLaSK"""

        remove_self = parent.document.filename is None and not parent.isWindowModified()

        iname = QFileDialog.getOpenFileName(parent, "Import EFM file", gui.CURRENT_DIR,
                                                  "EFM file (*.efm)")
        if type(iname) == tuple:
            iname = iname[0]
        if not iname:
            return
        dest_dir = os.path.dirname(iname)

        try:
            read = read_efm(iname)
            obase = os.path.join(dest_dir, os.path.basename(iname)[:-4])
        except Exception as err:
            if gui._DEBUG:
                import traceback
                traceback.print_exc()
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Import Error")
            msgbox.setText("There was an error while reading the EFM file.\n\n"
                           "Probably the chosen file was not in a EFM or the parser does not understand its syntax.")
            msgbox.setDetailedText(str(err))
            msgbox.setStandardButtons(QMessageBox.Ok)
            msgbox.setIcon(QMessageBox.Critical)
            msgbox.exec_()
        else:
            oname = obase + '.xpl'
            n = 1
            while os.path.exists(oname):
                oname = obase + '-{}.xpl'.format(n)
                n += 1
            try:
                write_xpl(oname, *read)
            except Exception as err:
                if gui._DEBUG:
                    import traceback
                    traceback.print_exc()
                msgbox = QMessageBox()
                msgbox.setWindowTitle("Import Error")
                msgbox.setText("There was an error while saving the converted XPL file.")
                msgbox.setDetailedText(str(err))
                msgbox.setStandardButtons(QMessageBox.Ok)
                msgbox.setIcon(QMessageBox.Critical)
                msgbox.exec_()
            else:
                new_window = gui.MainWindow(oname)
                try:
                    if new_window.document.filename is not None:
                        new_window.resize(parent.size())
                        gui.WINDOWS.add(new_window)
                        if remove_self:
                            parent.close()
                        else:
                            new_window.move(parent.x() + 24, parent.y() + 24)
                    else:
                        new_window.setWindowModified(False)
                        new_window.close()
                except AttributeError:
                    new_window.setWindowModified(False)
                    new_window.close()

    def import_efm_operation(parent):
        action = QAction(QIcon.fromTheme('document-open'),
                               'Import E&FM file...', parent)
        action.triggered.connect(lambda: import_efm(parent))
        return action


if __name__ == '__main__':
    iname = sys.argv[1]
    dest_dir = os.path.dirname(iname)
    read = read_efm(iname)
    obase = os.path.join(dest_dir,
                         os.path.basename(iname)[:-9] if iname[-9] == '_' else os.path.basename(iname)[:-4])

    oname = obase + '.xpl'
    n = 1
    while os.path.exists(oname):
        oname = obase + '-{}.xpl'.format(n)
        n += 1

    write_xpl(oname, *read[1:])

    print("Wrote '{}'".format(oname))
