# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import html
from copy import deepcopy
from lxml import etree

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

try:
    import cPickle as pickle
except ImportError:
    import pickle

from ...qt import QtSignal
from ...qt.QtCore import *
from ...qt.QtWidgets import *
from ...qt.QtGui import *

from .. import SectionModel, Info
from .reader import GNReadConf
from .constructor import construct_geometry_object, construct_by_name
from ..info import info_level_icon
from ...utils.xml import AttributeReader, OrderedTagReader
from .types import geometry_types_geometries, gname
from .node import GNFakeRoot
from .again_copy import GNAgain


basestring = str, bytes
class PyObjMime(QMimeData):
    MIMETYPE = 'application/x-pyobj'

    def __init__(self, data=None):
        super().__init__()
        self.data = data
        if data is not None:
            # Try to pickle data
            try:
                pdata = pickle.dumps(data)
            except:
                # We still can use self.data locally, but when fake_root is used this does not work
                self.setData(self.MIMETYPE, b'')
                return
            self.setData(self.MIMETYPE, pickle.dumps(data.__class__) + pdata)

    def itemInstance(self):
        if self.data is not None:
            return self.data
        io = StringIO(str(self.data(self.MIMETYPE)))
        try:
            # Skip the type.
            pickle.load(io)
            # Recreate the data.
            return pickle.load(io)
        except:
            pass

        return None


class GeometryModel(SectionModel, QAbstractItemModel):

    REMOVE_COMMAND_ID = 0

    dropped = QtSignal(QModelIndex)

    class RemoveChildrenCommand(QUndoCommand):

        def __init__(self, model, parent_node, row, end, parent=None):
            self.model = model
            self.parent_node = parent_node
            self.row = row
            self.end = end
            self.removed_elements = self.children_list[row:end]
            if len(self.removed_elements) > 1:
                name = 'items'
            else:
                name = gname(self.removed_elements[0].tag_name(full_name=False))
            super().__init__("remove "+name, parent)

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

        def id(self):
            return GeometryModel.REMOVE_COMMAND_ID

    class InsertChildCommand(QUndoCommand):

        def __init__(self, model, parent_node, row, child_node, merge_with_next_remove=False, parent=None, name="add"):
            self.model = model
            self.parent_node = parent_node
            if row is None: row = parent_node.new_child_pos()
            self.row = row
            self.child_node = child_node
            self.merge_with_next_remove = merge_with_next_remove
            self.next_remove = None
            super().__init__("{} {}".format(name, gname(child_node.tag_name(full_name=False))), parent)

        @property
        def parent_index(self):
            return self.model.index_for_node(self.parent_node)

        @property
        def children_list(self):
            return self.model.children_list(self.parent_node)

        def redo(self):
            self.model.beginInsertRows(self.parent_index, self.row, self.row)
            self.child_node.set_parent(self.parent_node, index=self.row, remove_from_old_parent=False)
            self.model.endInsertRows()
            if self.next_remove is not None:
                self.next_remove.redo()  # this also will call fire_changed()
            else:
                self.model.fire_changed()

        def undo(self):
            if self.next_remove is not None:
                self.next_remove.undo()
            self.model.beginRemoveRows(self.parent_index, self.row, self.row)
            del self.children_list[self.row]
            self.model.endRemoveRows()
            self.model.fire_changed()

        def mergeWith(self, command):
            if self.merge_with_next_remove:
                self.next_remove = command
                return True
            else:
                return False

        def id(self):
            if self.merge_with_next_remove and self.next_remove is None:
                return GeometryModel.REMOVE_COMMAND_ID
            return super(GeometryModel.InsertChildCommand, self).id()

    class SwapChildrenCommand(QUndoCommand):

        def __init__(self, model, parent_node, index1, index2, parent = None):
            if index2 < index1:
                self.index1, self.index2 = index2, index1
            else:
                self.index1, self.index2 = index1, index2
            if parent_node is None: parent_node = model.fake_root
            self.parent_node = parent_node
            self.model = model
            super(GeometryModel.SwapChildrenCommand, self).__init__('swap items of {} at rows {} and {}'
                                                                    .format(parent_node.tag_name(full_name=False),
                                                                            self.index1+1, self.index2+1), parent)

        def redo(self):
            parent_index = self.model.index_for_node(self.parent_node)
            self.model.beginMoveRows(parent_index, self.index2, self.index2, parent_index, self.index1)
            self.parent_node.children[self.index1], self.parent_node.children[self.index2] =\
                        self.parent_node.children[self.index2], self.parent_node.children[self.index1]
            self.model.endMoveRows()
            self.model.fire_changed()

        def undo(self):
            self.redo()

    class ReparentCommand(QUndoCommand):

        def __init__(self, model, parent_node, row, child_node, new_parent, parent=None):
            self.model = model
            self.parent_node = parent_node
            self.row = row
            self.child_node = child_node
            self.new_parent = new_parent
            new_child = deepcopy(child_node, memo={id(child_node._parent): child_node._parent})
            new_parent.path, new_parent.in_parent_aligners = new_child.path, new_child.in_parent_aligners
            new_child.path = new_child.in_parent_aligners = None
            new_child.set_parent(self.new_parent, self.new_parent.new_child_pos(), remove_from_old_parent=False)
            super(GeometryModel.ReparentCommand, self).__init__(
                "insert {} into {}".format(gname(child_node.tag_name(full_name=False)),
                                           gname(new_parent.tag_name(full_name=False))), parent)

        @property
        def parent_index(self):
            return self.model.index_for_node(self.parent_node)

        @property
        def children_list(self):
            return self.model.children_list(self.parent_node)

        def redo(self):
            parent_index = self.parent_index
            self.model.beginRemoveRows(parent_index, self.row, self.row)
            del self.children_list[self.row]
            self.model.endRemoveRows()
            self.model.beginInsertRows(parent_index, self.row, self.row)
            self.new_parent.set_parent(self.parent_node, index=self.row, remove_from_old_parent=False,
                                       check_parent_params=False)
            self.model.endInsertRows()
            self.model.fire_changed()

        def undo(self):
            parent_index = self.parent_index
            self.model.beginRemoveRows(parent_index, self.row, self.row)
            del self.children_list[self.row]
            self.model.endRemoveRows()
            self.model.beginInsertRows(parent_index, self.row, self.row)
            self.child_node.set_parent(self.parent_node, index=self.row, remove_from_old_parent=False)
            self.model.endInsertRows()
            self.children_list[self.row] = self.child_node
            self.model.fire_changed()

    class SetRootsCommand(QUndoCommand):

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

    def __init__(self, parent=None, info_cb=None):
        QAbstractItemModel.__init__(self, parent)
        SectionModel.__init__(self, 'geometry', info_cb)
        #TableModelEditMethods.__init__(self)
        self.fake_root = GNFakeRoot(self)
        self.axes = None    #TODO ? use axes of FakeRoot
        self._message = None
        self.dirty = False
        self.show_props = True
        self.endcomments = []
        self._moved_node = None

    @property
    def roots(self):
        return self.fake_root.children

    @roots.setter
    def roots(self, new_roots):
        self.fake_root.children = new_roots

    # XML element that represents whole section
    def make_xml_element(self):
        res = etree.Element(self.name)
        if self.axes: res.attrib['axes'] = self.axes
        conf = GNReadConf(axes=self.axes)
        for geom in self.roots:
            for c in geom.comments:
                res.append(etree.Comment(c))
            res.append(geom.make_xml_element(conf))
        for c in self.endcomments:
            res.append(etree.Comment(c))
        return res

    def load_xml_element(self, element, undoable=True):
        with AttributeReader(element) as a: new_axes = a.get('axes')
        conf = GNReadConf(axes=new_axes)
        new_roots = []
        with OrderedTagReader(element) as reader:
            for child_element in reader:
                root = construct_geometry_object(child_element, conf, geometry_types_geometries)
                root._parent = self.fake_root
                new_roots.append(root)
            self.endcomments = reader.get_comments()
        command = GeometryModel.SetRootsCommand(self, new_axes, new_roots)
        if undoable:
            self.undo_stack.push(command)
        else:
            command.redo()
            self.undo_stack.clear()
        self.dirty = True

    def get_text(self):
        element = self.make_xml_element()
        if len(element) == 0: return ""
        lines = etree.tostring(element, pretty_print=True, encoding='unicode').splitlines()[1:-1]
        return "\n".join(line[2:] for line in lines)

    def stubs(self):
        res = 'class GEO:\n    """PLaSK object containing the defined geometry objects."""\n'
        res += '\n'.join(s for s in (r.stub() for r in self.roots) if s)
        return res

    # QAbstractItemModel implementation:
    def columnCount(self, parent=QModelIndex()):
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        if role == Qt.ItemDataRole.DisplayRole:  # or role == Qt.ItemDataRole.EditRole:
            item = index.internalPointer()
            if index.column() == 0:
                return item.display_name(full_name=False)
            else:
                name = getattr(item, 'name', '')
                if self.show_props:
                    if name and not isinstance(item, GNAgain):
                        res = '<span style="color: #866">name:</span> <b>{}</b>'.format(html.escape(name))
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
                                res += '<span style="color: #769">[{}</span>'.format(html.escape(t).replace(' ', '&nbsp;'))
                                in_group = True
                            else:
                                n, v = t
                                if v is None: continue
                                if res: res += ' &nbsp;' if in_group else ' &nbsp; '
                                res += '<span style="color: #766">{}:</span>&nbsp;{}'\
                                    .format(html.escape(n).replace(' ', '&nbsp;'), html.escape(v).replace(' ', '&nbsp;'))
                                    # replacing ' ' to '&nbsp;' is for better line breaking (not in middle of name/value)
                elif name and not isinstance(item, GNAgain):
                    res = '<b>{}</b>'.format(html.escape(name))
                else:
                    res = ''
                return res
        elif role == Qt.ItemDataRole.ToolTipRole:
            item = index.internalPointer()
            res = '<table><tr><td colspan="2"><b>' + html.escape(item.display_name(full_name=False)) + '</b>'
            name = getattr(item, 'name', '')
            if name and not isinstance(item, GNAgain):
                res += '</td></tr><tr><td>name:</td><td><b>{}</b>'.format(html.escape(name))
            for prop_table in (item.in_parent_properties(), item.major_properties(), item.minor_properties()):
                in_group = False
                group_cont = False
                for t in prop_table:
                    if t is None:
                        if in_group:
                            res += "</td></tr></table>"
                            in_group = False
                    elif isinstance(t, basestring):
                        res += '</td></tr><tr><td>{}&nbsp;</td><td><table><tr><td>'\
                            .format(html.escape(t).replace(' ', '&nbsp;'))
                        in_group = True
                        group_cont = False
                    else:
                        n, v = t
                        if v is None: continue
                        if not in_group or group_cont:
                            res += '</td></tr><tr><td>'
                        else:
                            group_cont = True
                        res += '{}:&nbsp;</td><td>{}'\
                            .format(html.escape(n).replace(' ', '&nbsp;'), html.escape(v).replace(' ', '&nbsp;'))
                            # replacing ' ' to '&nbsp;' is for better line breaking (not in middle of name/value)
            res += '</td></tr></table>'
            return res
        elif role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            node = self.node_for_index(index)
            if node is self.fake_root: return
            if self._info is not None:
                for info in self._info:
                    try:
                        nodes = info.nodes
                    except AttributeError:
                        continue
                    else:
                        if node in nodes:
                            return info_level_icon(info.level)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags if self.is_read_only() else Qt.ItemFlag.ItemIsDropEnabled
        res = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not self.is_read_only():
            res |= Qt.ItemFlag.ItemIsDragEnabled
            if index.internalPointer().accept_new_child():
                res |= Qt.ItemFlag.ItemIsDropEnabled
        return res

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction | Qt.DropAction.CopyAction | Qt.DropAction.LinkAction

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return ('Tag', 'Properties' if self.show_props else 'Name')[section]
        return None

    def node_for_index(self, index):
        return index.internalPointer() if index.isValid() else self.fake_root

    def children_list(self, parent):
        """Get list of children of node or index."""
        if parent is None: return self.roots
        from .node import GNode
        if isinstance(parent, GNode): return parent.children
        return parent.internalPointer().children if parent.isValid() else self.roots

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent): return QModelIndex()
        l = self.children_list(parent)
        return self.createIndex(row, column, l[row]) #if 0 <= row < len(l) else QModelIndex()

    def parent(self, index):
        if not index.isValid(): return QModelIndex()
        return self.index_for_node(index.internalPointer().parent)
        #childItem = index.internalPointer()
        #parentItem = childItem.parent
        #if parentItem is None: return QModelIndex()
        #return self.createIndex(self.children_list(parentItem.parent).index(parentItem), 0, parentItem)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0: return 0
        return len(self.children_list(parent))

    def removeRows(self, row, count, parent=QModelIndex()):
        l = self.children_list(parent)
        end = row + count
        if row < 0 or end > len(l):
            return False
        self.undo_stack.push(
            GeometryModel.RemoveChildrenCommand(self, self.node_for_index(parent), row, end))
        return True

    def mimeTypes(self):
        return [PyObjMime.MIMETYPE]

    def mimeData(self, indexes):
        return PyObjMime(indexes[0].internalPointer())

    def fire_changed(self, *args, **kwargs):
        super().fire_changed(*args, **kwargs)
        if self._moved_node is not None:
            self.dropped.emit(self.index_for_node(self._moved_node))
            self._moved_node = None

    def dropMimeData(self, mime_data, action, row, column, parentIndex):
        if not self.canDropMimeData(mime_data, action, row, column, parentIndex):
            return False    # qt should call this earlier but some version of qt have a bug
        if action == Qt.DropAction.IgnoreAction: return True
        if action == Qt.DropAction.MoveAction:
            moved_obj = mime_data.itemInstance()
            parent = self.node_for_index(parentIndex)
            if moved_obj.parent != parent:
                # without copy, the parent of source is incorrect after changing and then remove will crash
                moved_obj = deepcopy(moved_obj, memo={id(moved_obj._parent): moved_obj._parent})
            self.insert_node(parent, moved_obj, None if row == -1 else row, merge_with_next_remove=True, action_name="move")
            self._moved_node = moved_obj
            return True # removeRows will be called and will remove current moved_obj
        if action == Qt.DropAction.CopyAction:
            copied_obj = mime_data.itemInstance()
            parent = self.node_for_index(parentIndex)
            copied_obj = deepcopy(copied_obj, memo={id(copied_obj._parent): copied_obj._parent})
            try:
                if copied_obj.name is not None: copied_obj.name = None
            except AttributeError:
                pass
            self.insert_node(parent, copied_obj, None if row == -1 else row, action_name="copy")
            self.dropped.emit(self.index_for_node(copied_obj))
            return True
        if action == Qt.DropAction.LinkAction:
            linked_obj = mime_data.itemInstance()
            parent = self.node_for_index(parentIndex)
            from .again_copy import GNAgain
            again = GNAgain(ref=linked_obj.name)
            self.insert_node(parent, again, None if row == -1 else row, action_name="make")
            self.dropped.emit(self.index_for_node(again))
            return True
        return False

    def canDropMimeData(self, mime_data, action, row, column, parentIndex):
        try:
            if action in (Qt.DropAction.MoveAction, Qt.DropAction.CopyAction):
                moved_obj = mime_data.itemInstance()
                parent = parentIndex.internalPointer()
                if parent is None:
                    from .geometry import GNGeometryBase
                    return isinstance(moved_obj, GNGeometryBase)
                else:
                    return moved_obj not in parent.path_to_root and parent.accept_as_child(moved_obj)
            if action == Qt.DropAction.LinkAction:
                linked_obj = mime_data.itemInstance()
                parent = parentIndex.internalPointer()
                if parent is None or linked_obj.name is None:
                    return False
                from .again_copy import GNAgain
                return parent.accept_as_child(GNAgain())  # and linked_obj.name in self.names_before(parent)
        except:
            pass
        return False

    #def moveRow(sourceParent, sourceRow, destinationParent, destinationChild):
    #    return False

    #def insertRows(self, row, count, parent = QModelIndex()):
    #    pass

    def index_for_node(self, node, column=0):
        if node is None or isinstance(node, GNFakeRoot): return QModelIndex()
        try:
            c = node.parent.children if node.parent else self.roots
            index = self.createIndex(c.index(node), column, node)
        except (ValueError, IndexError, TypeError):
            return QModelIndex()
        return index

    def insert_node(self, parent_node, child_node, pos=None, merge_with_next_remove=False, action_name="add"):
        self.undo_stack.push(
            GeometryModel.InsertChildCommand(self, parent_node, pos, child_node, merge_with_next_remove=merge_with_next_remove,
                                             name=action_name))

    def append_geometry(self, type_name):
        self.insert_node(self.fake_root, construct_by_name(type_name, geometry_types_geometries), len(self.roots))

    def _swap_neighbour_nodes(self, parent_index, row1, row2):
        if self.is_read_only(): return
        if row2 < row1: row1, row2 = row2, row1
        children = self.children_list(parent_index)
        if row1 < 0 or row2 >= len(children): return
        self.undo_stack.push(
            GeometryModel.SwapChildrenCommand(self, self.node_for_index(parent_index), row1, row2)
        )

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

    def duplicate(self, index):
        item = index.internalPointer()
        parent = index.parent()
        parent_node = self.node_for_index(parent)
        if not parent_node.accept_new_child(): return
        copied = deepcopy(item, memo={id(item._parent): item._parent})
        try:
            if copied.name is not None: copied.name = None
        except AttributeError:
            pass
        self.insert_node(parent_node, copied, action_name="duplicate")
        return self.index_for_node(copied)

    def reparent(self, index, type_constructor):
        parent_index = index.parent()
        parent_node = self.node_for_index(parent_index)
        row = index.row()
        child_node = index.internalPointer()
        new_parent = type_constructor(None, None)
        self.undo_stack.push(
            GeometryModel.ReparentCommand(self, parent_node, row, child_node, new_parent))
        return self.index_for_node(new_parent)

    def get_names_before(self, end_node):
        res = set()
        for r in self.roots:
            if not r.get_names_before(res, end_node): break
        return res

    def get_names(self, filter=None):
        res = set()
        for r in self.roots: res |= r.get_names(filter)
        return res

    def get_paths(self, filter=None):
        res = set()
        for r in self.roots: res |= r.get_paths(filter)
        return res

    def find_by_name(self, name):
        for r in self.roots:
            found = r.find_by_name(name)
            if found is not None:
                return found

    @property
    def roots_cartesian2d(self):
        return (root for root in self.roots if isinstance(root, GNCartesian) and root.dim == 2)

    @property
    def roots_cylindrical(self):
        return (root for root in self.roots if isinstance(root, GNCylindrical))

    @property
    def roots_cartesian3d(self):
        return (root for root in self.roots if isinstance(root, GNCartesian) and root.dim == 3)

    def get_roots(self, dim=None):
        if dim is None:
            return self.roots
        else:
            return (root for root in self.roots if root.dim == dim)

    def create_info(self):
        res = super().create_info()
        names = {}
        for root in self.roots:
            for node in root.traverse_dfs():
                node.create_info(res, names)
            if root.name is None:
                res.append(Info('{} geometry does not have a name.'.format(gname(root.tag_name(full_name=False))),
                                Info.ERROR, nodes=(root,), property='name'))
        for name, nodes in names.items():
            if len(nodes) > 1:
                res.append(Info('{} objects have the same name "{}".'.format(len(nodes), name),
                                Info.ERROR, nodes=nodes, property='name'))
        if self._message is not None:
            if self._message['level'] >= Info.WARNING:
                res.insert(0, Info(**self._message))
            else:
                res.append(Info(**self._message))
        return res

    def info_message(self, msg=None, level=Info.INFO, **kwargs):
        if msg is None:
            self._message = None
        else:
            self._message = dict(text=msg, level=level)
            self._message.update(kwargs)
        self.refresh_info()
        self.fire_info_changed()


from .geometry import GNCartesian, GNCylindrical
