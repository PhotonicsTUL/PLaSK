#!/usr/bin/python
#coding: utf8

from warnings import warn
from numpy import sqrt, all

try:
    from lxml import etree
except ImportError:
    from xml.etree import cElementTree as etree

import pslab as ps
from pslab import Geometry as pg


class _UniqueId(object):
    """Generator of unique names"""

    def __init__(self, prefix, suffix='', fmt="d", initial=0):
        self.prefix = prefix
        self.suffix = suffix
        self.fmt = fmt
        self.counter = initial - 1

    def __call__(self):
        self.counter += 1
        return self.prefix + ('{:'+self.fmt+'}').format(self.counter) + self.suffix


def _iter_objects(geo):
    for lay in set(l for l,h in geo.layers):
        for o in lay.objects:
            yield o


def _convert_rectangle(obj, h):
    item = etree.Element("item")
    item.attrib['left'] = str(obj.left)
    rectangle = etree.SubElement(item, "rectangle")
    rectangle.attrib['dy'] = str(obj.right-obj.left)
    rectangle.attrib['dz'] = str(h)
    return item, rectangle

def _convert_cuboid(obj, h):
    item = etree.Element("item")
    item.attrib['back'], item.attrib['left'] = map(str, obj.corner1)
    cuboid = etree.SubElement(item, "cuboid")
    cuboid.attrib['dx'], cuboid.attrib['dy'] = map(str, obj.corner2 - obj.corner1)
    cuboid.attrib['dz'] = str(h)
    return item, cuboid

def _convert_cylinder(obj, h):
    item = etree.Element("item")
    item.attrib['x'], item.attrib['y'] = map(str, obj.center)
    cylinder = etree.SubElement(item, "cylinder")
    cylinder.attrib['radius'] = str(obj.radius)
    cylinder.attrib['height'] = str(h)
    return item, cylinder

CONVERTERS = {
    pg.Rectangle: _convert_rectangle,
    pg.Cuboid: _convert_cuboid,
    pg.Cylinder: _convert_cylinder,
    pg.PML: None,
    pg.PMLs: None,
}


def export_geometry(geo, materials=None):
    """
    Export PSlab geometry as PLaSK XML element.

    Args:
        geo: PSlab geometry object.
        materials (dict): Dictionary mapping epsilon to material names

    Returns:
        etree.Element: Root XML element <plask>
    """

    material_names = {}
    if materials is not None:
        material_names.update(dict((str(k), v) for k,v in materials.items()))

    root = etree.Element("plask")
    materials = etree.SubElement(root, "materials")
    geometry = etree.SubElement(root, "geometry")

    if isinstance(geo, pg.Geometry2D):
        geometry = etree.SubElement(geometry, "cartesian2d")
        geometry.attrib['axes'] = "yz"
        geometry.attrib['name'] = "pslab"
        laya = {'dy': geo.args[0]}
        has_pml = False
        for o in _iter_objects(geo):
            if isinstance(o, pg.PML):
                has_pml = True
                laya['dy'] -= o.size
                break
        geometry.attrib['top'] = geometry.attrib['bottom'] = "extend"
        geometry.attrib['right'] = "extend" if has_pml else "periodic"
        if geo.symmetry is not None:
            geometry.attrib['left'] = "mirror"
            geometry = etree.SubElement(geometry, "clip")
            geometry.attrib['left'] = "0."
        elif has_pml:
            geometry.attrib['left'] = "extend" if has_pml else "periodic"
        stack = etree.SubElement(geometry, "stack")
        align = {'y': "0."}
        lalign = {'ycenter': "0."}

    elif isinstance(geo, pg.Geometry3D):
        if (geo.A1[1] != 0. or geo.A2[0] != 0.) and (geo.A1[0] != 0. or geo.A2[1] != 0.):
            raise TypeError("non-rectangular geometries not supported")
        laya = {'dx': geo.A1[0] + geo.A2[0], 'dy': geo.A1[1] + geo.A2[1]}
        geometry = etree.SubElement(geometry, "cartesian3d")
        geometry.attrib['name'] = "pslab"
        geometry.attrib['axes'] = "xyz"
        has_pml = False
        for o in _iter_objects(geo):
            if isinstance(o, pg.PMLs):
                has_pml = True
                s = o.size
                if isinstance(s, float): s = s, s
                laya['dx'] -= s[0]
                laya['dy'] -= s[1]
                break
        geometry.attrib['top'] = geometry.attrib['bottom'] = "extend"
        geometry.attrib['right'] = geometry.attrib['front'] = "extend" if has_pml else "periodic"
        if isinstance(geo.symmetry, (tuple, list)):
            sym = tuple(geo.symmetry)
        else:
            sym = geo.symmetry, geo.symmetry
        geometry.attrib['back'] = "mirror" if sym[0] is not None else "extend" if has_pml else "periodic"
        geometry.attrib['left'] = "mirror" if sym[1] is not None else "extend" if has_pml else "periodic"
        if sym[0] is not None or sym[1] is not None:
            geometry = etree.SubElement(geometry, "clip")
            if sym[0] is not None: geometry.attrib['back'] = "0."
            if sym[1] is not None: geometry.attrib['left'] = "0."
        stack = etree.SubElement(geometry, "stack")
        align = {'x': "0.", 'y': "0."}
        lalign = {'xcenter': "0.", 'ycenter': "0."}

    else:
        raise TypeError('Wrong PSlab geometry type')

    laya = dict((k, str(v)) for k,v in laya.items())
    stack.attrib.update(align)

    new_material_name = _UniqueId('material', fmt='02d')
    new_layer_name = _UniqueId('layer')

    previous_layers = {}

    def parse_material(obj):
        eps = obj.eps
        if all(eps == eps[0]): eps = eps[0]
        if eps == 1.:
            return 'air'
        if eps.imag == 0:
            eps = eps.real
        key = str(eps)
        mat = material_names.get(key)
        if mat is None:
            new_material = etree.Element("material")
            mat = new_material_name()
            new_material.attrib['name'] = material_names[key] = mat
            new_material.attrib['base'] = "semiconductor"
            if isinstance(eps, (float, complex)):
                nr = etree.SubElement(new_material, "Nr")
                n = sqrt(eps)
                if n.imag == 0: nr.text = str(n.real)
                else: nr.text = str(n)[1:-1]
            else:
                nr = etree.SubElement(new_material, "NR")
                nr.text = ', '.join(str(sqrt(e)) for e in eps)
            materials.append(new_material)
        return mat

    layers, heights, idxs = geo.generateStack(True)
    for i,h in zip(idxs, heights):
        layer = previous_layers.get((i,h))
        if layer is None:
            if not layers[i].objects:
                item = etree.SubElement(stack, "item")
                item.attrib.update(lalign)
                layer = etree.SubElement(item, "block")
                layer.attrib['dz'] = str(h)
                layer.attrib.update(laya)
                layer.attrib['material'] = parse_material(layers[i])
            else:
                layer = etree.SubElement(stack, "align")
                layer.attrib['z'] = "0."
                layer.attrib.update(align)
                item = etree.SubElement(layer, "item")
                item.attrib.update(lalign)
                block = etree.SubElement(item, "block")
                block.attrib['dz'] = str(h)
                block.attrib.update(laya)
                block.attrib['material'] = parse_material(layers[i])
                for obj in layers[i].objects:
                    try:
                        conv = CONVERTERS[type(obj)]
                    except KeyError:
                        warn("Object type not recognized: {s}".format(type(obj)))
                    else:
                        if conv is None: continue
                        item, leaf = conv(obj, h)
                        leaf.attrib['material'] = parse_material(obj)
                        layer.append(item)
            previous_layers[(i,h)] = layer
        else:
            if 'name' not in layer.attrib:
                layer.attrib['name'] = new_layer_name()
            item = etree.SubElement(stack, "item")
            item.attrib.update(lalign)
            again = etree.SubElement(item, "again")
            again.attrib['ref'] = layer.attrib['name']

    return root


def export_simulation(sim, materials=None):
    """
    Export PSlab simulation as PLaSK XML element.

    Args:
        sim: PSlab simulation object.
        materials (dict): Dictionary mapping epsilon to material names

    Returns:
        etree.Element: Root XML element <plask>
    """
    xpl = export_geometry(sim.geometry, materials)
    #TODO
    return xpl


def save_xpl(xplname, source, materials=None):
    """
    Export PSlab geometry as PLaSK geometry XML element.

    Args:
        xplname (str): XPL filename
        source: PSlab simulation or geometry object.
        materials (dict): Dictionary mapping epsilon to material names
    """

    if isinstance(source, pg.Geometry):
        xpl = export_geometry(source, materials)
    elif isinstance(source, ps._SlabSimulation):
        xpl = export_simulation(source, materials)
    else:
        raise TypeError('Wrong PSLab source type')

    with open(xplname, 'w') as out:
        out.write(etree.tostring(xpl, encoding='unicode', pretty_print=True))
