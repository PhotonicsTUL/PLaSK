# coding: utf8

from xml.etree import ElementTree
from PyQt4 import QtGui
from qhighlighter.XML import XMLHighlighter

class SectionModel(object):

    def __init__(self, tagName):
        object.__init__(self)
        self.tagname = tagName

    def getText(self):
        element = self.getXMLElement()
        text = ''
        #text = ''+element.text
        for c in element:
            text += ElementTree.tostring(c)
        if not text:
            return ''
        return text
        #return ElementTree.tostring(self.getXMLElement())

    def setText(self, text):
        self.setXMLElement(ElementTree.fromstringlist(['<?xml version="1.0"?>\n<', self.tagname, '>', text, '</', self.tagname, '>']))

    def createSourceEditor(self):
        ed = QtGui.QTextEdit()
        self.highlighter = XMLHighlighter(ed.document())   # highlighter varible is required, in other case it is deleted and text is not highlighted
        return ed

    # text, source editor
    def getSourceEditor(self):
        if not hasattr(self, 'sourceEditor'): self.sourceEditor = self.createSourceEditor()
        self.sourceEditor.setPlainText(self.getText())
        return self.sourceEditor

    # GUI editor, by default use source editor
    def getEditor(self):
        return self.getSourceEditor()

    # when editor is turn off, model should be update
    def afterEdit(self, editor):
        self.setText(self.getSourceEditor().toPlainText()) #TODO getSourceEditor() nadpisze zawartość edytora, bez sensu


class SectionModelTreeBased(SectionModel):

    def __init__(self, tagName):
        SectionModel.__init__(self, tagName)
        self.element = ElementTree.Element(tagName)

    def setXMLElement(self, element):
        if isinstance(element, ElementTree.Element):
            self.element = element
        else:
            self.element.clear()

    # XML element that represents whole section
    def getXMLElement(self):
        return self.element



