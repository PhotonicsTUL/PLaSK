from ..qt import QtGui

class UndoCommandWithSetter(QtGui.QUndoCommand):
    """
        Undo command which change node using setter method, to new_value and call model.fire_changed after each change.
        Node can optionally be equal to model.
    """

    def __init__(self, model, node, setter, new_value, old_value, action_name, QUndoCommand_parent = None):
        super(UndoCommandWithSetter, self).__init__(action_name, QUndoCommand_parent)
        self.model = model
        self.node = node
        self.setter = setter
        self.new_value = new_value
        self.old_value = old_value

    def set_property_value(self, value):
        self.setter(self.node, value)
        self.model.fire_changed()

    def redo(self):
        self.set_property_value(self.new_value)

    def undo(self):
        self.set_property_value(self.old_value)
