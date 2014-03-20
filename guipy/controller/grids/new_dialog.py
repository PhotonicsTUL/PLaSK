from PyQt4 import QtGui
from PyQt4.Qt import QLabel, QStringListModel
from model.grids.grid import Grid
from model.grids.types import construct_grid , meshes_types, generators_types,\
    generator_methods
 
class NewGridDialog(QtGui.QDialog):
 
    def __init__(self, parent=None):
        super(NewGridDialog, self).__init__(parent)
        self.setWindowTitle("New grid")
        
        kind = QtGui.QGroupBox("Kind")
        self.kind_mesh = QtGui.QRadioButton("&Mesh")
        self.kind_generator = QtGui.QRadioButton("&Generator")
        self.kind_generator.toggled.connect(self.__set_mode__)
        self.kind_mesh.setChecked(True)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.kind_mesh)
        hbox.addWidget(self.kind_generator)
        #hbox.addStretch(1)    #??
        kind.setLayout(hbox)
        
        self.name_edit = QtGui.QLineEdit()
        
        self.type_edit = QtGui.QComboBox()
        self.type_edit.setEditable(True)
        self.type_edit.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.type_edit.editTextChanged.connect(self.__type_changed__)
        
        self.method_edit = QtGui.QComboBox()
        self.method_edit.setEditable(True)
        self.method_edit.setInsertPolicy(QtGui.QComboBox.NoInsert)
        self.method_edit_label = QLabel("Method:")
        
        formLayout = QtGui.QFormLayout();
        formLayout.addRow("Name:", self.name_edit)
        formLayout.addRow("Type:", self.type_edit)
        formLayout.addRow(self.method_edit_label, self.method_edit)
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui. QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(kind)
        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)
        
        self.__set_mode__(False)
        
    def __set_mode__(self, is_generator):
        self.method_edit.setVisible(is_generator)
        self.method_edit_label.setVisible(is_generator)
        #self.method_edit.setEnabled(is_generator)
        #self.type_edit.clear()
        text = self.type_edit.currentText()
        if is_generator:
            self.type_edit.setModel(QStringListModel(sorted(generators_types())))
        else:
            self.type_edit.setModel(QStringListModel(sorted(meshes_types())))
        self.type_edit.setEditText(text)
        self.__type_changed__(text)
        
    def __type_changed__(self, new_type):
        if not self.method_edit.isVisible(): return
        text = self.method_edit.currentText()
        self.method_edit.setModel(QStringListModel(sorted(generator_methods(new_type))))
        self.method_edit.setEditText(text)
    
    def get_grid(self, grids_model):
        return construct_grid(
                grids_model,
                Grid.contruct_empty_XML_element(
                    self.name_edit.text(),
                    self.type_edit.currentText(),
                    self.method_edit.currentText() if self.kind_generator.isChecked() else None
                )
        )
    

def construct_grid_using_dialog(grids_model):
    dial = NewGridDialog()
    if dial.exec_() == QtGui.QDialog.Accepted:
        return dial.get_grid(grids_model)
    return None