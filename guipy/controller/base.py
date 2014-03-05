class Controller(object):
    
    def __init__(self, document, model):
        object.__init__(self)
        self.document = document
        self.model = model
        
    def saveDataInModel(self):
        """Called to force save data from editor in model."""
        pass  
        
    def onEditEnter(self):
        """Called when editor is entered and will be visible."""
        pass

    def onEditExit(self):
        """Called when editor is left and will be not visible. Typically and by default it calls saveDataInModel."""
        self.saveDataInModel()       
