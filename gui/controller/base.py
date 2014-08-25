from ..qt import QtGui, QtCore

class Controller(object):
    """
        Base class for controllers.
        Controllers create editor for the fragment of the XPL document (a section or smaller fragment) and make it
        available by get_editor method (subclasses must implement this method).
        They also transfer data from model to editor (typically in on_edit_enter) and in opposite direction
        (typically in save_data_in_model).
    """
    
    def __init__(self, document=None, model=None):
        """Optionally set document and/or model."""
        super(Controller, self).__init__()
        if document is not None: self.document = document
        if model is not None:
            self.model = model

        
    def save_data_in_model(self):
        """Called to force save data from editor in model (typically by on_edit_exit or when model is needed while
         editor still is active - for instance when user saves edited document to file)."""
        pass  
        
    def on_edit_enter(self):
        """Called when editor is entered and will be visible."""
        pass

    def on_edit_exit(self):
        """Called when editor is left and will be not visible. Typically and by default it calls save_data_in_model."""
        self.save_data_in_model()
        if hasattr(self, 'document'): self.document.window.set_section_actions()

    # def get_editor(self) - to be done in subclasses



class NoConfController(Controller):
    """Controller for all models that does not require any configuration."""
    def __init__(self, text = 'Configuration is neither required nor available.'):
        super(NoConfController, self).__init__()
        self.label = QtGui.QLabel(text)
        self.label.setAlignment(QtCore.Qt.AlignCenter)

    def get_editor(self):
        return self.label
