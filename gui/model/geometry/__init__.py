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
import cgi

from lxml import etree
from copy import copy
import operator
import cStringIO
try:
    import cPickle as pickle
except ImportError:
    import pickle

from ...qt import QtCore, QtGui

from .. import SectionModel
from .reader import GNReadConf
from .constructor import construct_geometry_object, construct_by_name
from ...utils.xml import AttributeReader
from .types import geometry_types_geometries

__author__ = 'qwak'


class PyObjMime(QtCore.QMimeData):
    MIMETYPE = 'application/x-pyobj'

    def __init__(self, data=None):
        super(PyObjMime, self).__init__()
        self.data = data
        if data is not None:
            # Try to pickle data
            try:
                pdata = pickle.dumps(data)
            except:
                return
            self.setData(self.MIMETYPE, pickle.dumps(data.__class__) + pdata)

    def itemInstance(self):
        if self.data is not None:
            return self.data
        io = cStringIO.StringIO(str(self.data(self.MIMETYPE)))
        try:
            # Skip the type.
            pickle.load(io)
            # Recreate the data.
            return pickle.load(io)
        except:
            pass

        return None


class GeometryModel(QtCore.QAbstractItemModel, SectionModel):

    def __init__(self, parent=None, info_cb=None):
        QtCore.QAbstractItemModel.__init__(self, parent)
        SectionModel.__init__(self, 'geometry', info_cb)
        #TableModelEditMethods.__init__(self)
        self.roots = []
        self.axes = None

    # XML element that represents whole section
    def get_xml_element(self):
        res = etree.Element(self.name)
        if self.axes: res.attrib['axes'] = self.axes
        conf = GNReadConf(axes=self.axes)
        for geom in self.roots: res.append(geom.get_xml_element(conf))
        return res

    class SetRootsCommand(QtGui.QUndoCommand):

        def __init__(self, model, axes, roots, QUndoCommand_parent = None):
            super(GeometryModel.SetRootsCommand, self).__init__('edit XPL source', QUndoCommand_parent)
            self.model = model
            self.old_axes = model.axes
            self.old_roots = model.roots
            self.new_axes = axes
            self.new_roots = roots

        def _set(self, axes, roots):
            self.model.beginResetModel()
            self.model.axes = axes
            self.model.roots = roots
            self.model.endResetModel()
            self.model.fire_changed()

        def redo(self):
            self._set(self.new_axes, self.new_roots)

        def undo(self):
            self._set(self.old_axes, self.old_roots)

    def set_xml_element(self, element, undoable=True):
        with AttributeReader(element) as a: new_axes = a.get('axes')
        conf = GNReadConf(axes=new_axes)
        new_roots = []
        for child_element in element:
            new_roots.append(construct_geometry_object(child_element, conf, geometry_types_geometries))
        command = GeometryModel.SetRootsCommand(self, new_axes, new_roots)
        if undoable:
            self.undo_stack.push(command)
        else:
            command.redo()
            self.undo_stack.clear()

    def stubs(self):
        res = 'class GEO(object):\n    """PLaSK object containing the defined geometry objects."""\n'
        res += '\n'.join(s for s in (r.stub() for r in self.roots) if s)
        return res

    # QAbstractItemModel implementation:
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 2

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role == QtCore.Qt.DisplayRole: #or role == QtCore.Qt.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                return item.display_name(full_name=False)
            else:
                name = getattr(item, 'name', '')
                if name:
                    res = '<span style="color: #866">name:</span> <b>{}</b>'.format(cgi.escape(name))
                else:
                    res = ''
                for prop_table in (item.in_parent_properties(), item.major_properties(), item.minor_properties()):
                    in_group = False
                    for t in prop_table:
                        if t is None:
                            if in_group: res += '<span style="color: #769">]</span>'
                            in_group = False
                        elif isinstance(t, basestring):
                            if res: res += ' &nbsp; '
                            res += '<span style="color: #769">[{}</span>'.format(cgi.escape(t).replace(' ', '&nbsp;'))
                            in_group = True
                        else:
                            n, v = t
                            if v is None: continue
                            if res: res += ' &nbsp;' if in_group else ' &nbsp; '
                            res += '<span style="color: #766">{}:</span>&nbsp;{}'\
                                .format(cgi.escape(n).replace(' ', '&nbsp;'), cgi.escape(v).replace(' ', '&nbsp;'))
                        # replacing ' ' to '&nbsp;' is for better line breaking (not in middle of name/value)
                return res

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags if self.is_read_only() else QtCore.Qt.ItemIsDropEnabled
        res = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if not self.is_read_only():
            res |= QtCore.Qt.ItemIsDragEnabled
            if index.internalPointer().accept_new_child():
                   res |= QtCore.Qt.ItemIsDropEnabled
        return res

    def supportedDropActions(self):
        return QtCore.Qt.MoveAction

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return ('tag', 'properties')[section]
        return None

    def children_list(self, parent):
        '''Get list of children of node or index.'''
        if parent is None: return self.roots
        from .node import GNode
        if isinstance(parent, GNode): return parent.children
        return parent.internalPointer().children if parent.isValid() else self.roots

    def index(self, row, column, parent = QtCore.QModelIndex()):
        if not self.hasIndex(row, column, parent): return QtCore.QModelIndex()
        l = self.children_list(parent)
        return self.createIndex(row, column, l[row]) #if 0 <= row < len(l) else QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QtCore.QModelIndex()
        return self.index_for_node(index.internalPointer().parent)
        #childItem = index.internalPointer()
        #parentItem = childItem.parent
        #if parentItem is None: return QtCore.QModelIndex()
        #return self.createIndex(self.children_list(parentItem.parent).index(parentItem), 0, parentItem)

    def rowCount(self, parent = QtCore.QModelIndex()):
        if parent.column() > 0: return 0
        return len(self.children_list(parent))

    class RemoveChildrenCommand(QtGui.QUndoCommand):

        def __init__(self, model, parent_node, row, end, QUndoCommand_parent=None):
            self.model = model
            self.parent_node = parent_node
            self.row = row
            self.end = end
            self.removed_elements = self.children_list[row:end]
            if len(self.removed_elements) > 1:
                name = 'items'
            else:
                name = self.removed_elements[0].tag_name(full_name=False)
            super(GeometryModel.RemoveChildrenCommand, self).__init__('remove {}'.format(name), QUndoCommand_parent)

        @property
        def parent_index(self):
            return self.model.index_for_node(self.parent_node)

        @property
        def children_list(self):
            return self.model.children_list(self.parent_node)

        def redo(self):
            self.model.beginRemoveRows(self.parent_index, self.row, self.end-1)
            del self.children_list[self.row:self.end]
            self.model.endRemoveRows()
            self.model.fire_changed()

        def undo(self):
            self.model.beginInsertRows(self.parent_index, self.row, self.end-1)
            self.children_list[self.row:self.row] = self.removed_elements
            self.model.endInsertRows()
            self.model.fire_changed()

    def removeRows(self, row, count, parent=QtCore.QModelIndex()):
        l = self.children_list(parent)
        end = row + count
        if row < 0 or end > len(l):
            return False
        self.undo_stack.push(
            GeometryModel.RemoveChildrenCommand(
                self, parent.internalPointer() if parent.isValid() else None, row, end))
        return True

    def mimeTypes(self):
        return [PyObjMime.MIMETYPE]

    def mimeData(self, indexes):
        return PyObjMime(indexes[0].internalPointer())

    def dropMimeData(self, mime_data, action, row, column, parentIndex):
        if action == QtCore.Qt.IgnoreAction: return True
        if action == QtCore.Qt.MoveAction:
            moved_obj = mime_data.itemInstance()
            parent = parentIndex.internalPointer()  # this can be None for root
            destination_list = self.children_list(parentIndex)
            if parent is None:
                from .geometry import GNGeometryBase
                if not isinstance(moved_obj, GNGeometryBase): return False
                if row == -1: row = len(destination_list)
            else:
                if moved_obj in parent.path_to_root or not parent.accept_as_child(moved_obj): return False
                if row == -1: row = parent.new_child_pos()
            self.beginInsertRows(parentIndex, row, row)
            #self.set_parent(parent, remove_from_old_parent_children=False, try_prevent_in_parent_params=True)
            # TODO problem: without copy, the parent of source is incorrect after change and remove crash
            moved_obj = copy(moved_obj) #TODO !!!! what with moved_obj children parent attribute?!
            if type(parent) != type(moved_obj._parent): moved_obj.in_parent = None
            from .container import GNContainerBase
            if not isinstance(parent, GNContainerBase): moved_obj.path = None
            #if moved_obj._parent != parent: moved_obj.path = None  # maybe this is better strategy?
            moved_obj._parent = parent
            destination_list.insert(row, moved_obj)
            self.endInsertRows()
            return True #removeRows will be called and remove current moved_obj
        return False

    def canDropMimeData(self, mime_data, action, row, column, parentIndex): #TODO this is optional but why qt doesn't call this??
    #     #return super(GeometryModel, self).canDropMimeData(mime_data, action, row, column, parentIndex)
        if action == QtCore.Qt.MoveAction:
            moved_obj = mime_data.itemInstance()
            parent = parentIndex.internalPointer()
            if parent is None:
                from .geometry import GNGeometryBase
                return isinstance(moved_obj, GNGeometryBase)
            else:
                return moved_obj not in parent.path_to_root and parent.accept_as_child(moved_obj)
        return False

    #def moveRow(sourceParent, sourceRow, destinationParent, destinationChild):
    #    return False

    #def insertRows(self, row, count, parent = QtCore.QModelIndex()):
    #    pass

    # other actions:
    def index_for_node(self, node):
        if node is None: return QtCore.QModelIndex()
        c = node.parent.children if node.parent else self.roots
        return self.createIndex(c.index(node), 0, node)

    def append_geometry(self, type_name):
        self.beginInsertRows(QtCore.QModelIndex(), len(self.roots), len(self.roots))
        self.roots.append(construct_by_name(type_name, geometry_types_geometries))
        self.endInsertRows()
        self.fire_changed()

    def _swap_neighbour_nodes(self, parent_index, row1, row2):
        if self.is_read_only(): return
        if row2 < row1: row1, row2 = row2, row1
        children = self.children_list(parent_index)
        if row1 < 0 or row2 >= len(children): return
        self.beginMoveRows(parent_index, row2, row2, parent_index, row1)
        children[row1], children[row2] = children[row2], children[row1]
        self.endMoveRows()
        self.fire_changed()

    def move_node_up(self, index):
        if not index.isValid(): return
        r = index.row()
        self._swap_neighbour_nodes(index.parent(), r-1, r)

    def move_node_down(self, index):
        if not index.isValid(): return
        r = index.row()
        self._swap_neighbour_nodes(index.parent(), r, r+1)

    def can_move_node_up_down(self, index):
        if not index.isValid(): return False, False
        children = self.children_list(index.parent())
        r = index.row()
        return r > 0, r+1 < len(children)

    def names_before(self, end_node):
        res = set()
        for r in self.roots:
            if not r.names_before(res, end_node): break
        return res

    def paths(self):
        res = set()
        for r in self.roots: res |= r.paths()
        return res