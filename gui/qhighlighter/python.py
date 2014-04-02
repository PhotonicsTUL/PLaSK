#!/usr/bin/env python

"""
highlightedtextedit.py

A PyQt custom widget example for Qt Designer.

Copyright (C) 2006 David Boddie <david@boddie.org.uk>
Copyright (C) 2005-2006 Trolltech ASA. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

http://www.henning-schroeder.de/codeaide/
"""

from PyQt4.QtCore import QRegExp
from PyQt4.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from PyQt4 import QtCore, QtGui

class PythonHighlighter(QSyntaxHighlighter):

    keywords = (
        "and",       "del",       "for",       "is",        "raise",
        "assert",    "elif",      "from",      "lambda",    "return",
        "break",     "else",      "global",    "not",       "try",
        "class",     "except",    "if",        "or",        "while",
        "continue",  "exec",      "import",    "pass",      "yield",
        "def",       "finally",   "in",        "print",     "with"
        )
    builtins = ('ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException', 'BufferError', 'BytesWarning', 
    'DeprecationWarning', 'EOFError', 'Ellipsis', 'EnvironmentError', 'Exception', 'False', 'FloatingPointError', 'FutureWarning', 
    'GeneratorExit', 'IOError', 'ImportError', 'ImportWarning', 'IndentationError', 'IndexError', 'KeyError', 'KeyboardInterrupt', 
    'LookupError', 'MemoryError', 'NameError', 'None', 
    'NotImplemented', 'NotImplementedError', 'OSError', 'OverflowError', 'PendingDeprecationWarning', 'ReferenceError', 'RuntimeError', 'RuntimeWarning', 
    'StandardError', 'StopIteration', 'SyntaxError', 'SyntaxWarning', 'SystemError', 'SystemExit', 'TabError', 'True', 'TypeError', 'UnboundLocalError', 
    'UnicodeDecodeError', 'UnicodeEncodeError', 'UnicodeError', 'UnicodeTranslateError', 'UnicodeWarning', 
    'UserWarning', 'ValueError', 'Warning', 'ZeroDivisionError', '__debug__', '__doc__', '__import__', '__name__', '__package__', 
    'abs', 'all', 'any', 'apply', 'basestring', 'bin', 'bool', 'buffer', 'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 
    'cmp', 'coerce', 'compile', 'complex', 'copyright', 'credits', 'delattr', 'dict', 'dir', 'divmod', 
    'enumerate', 'eval', 'execfile', 'exit', 'file', 'filter', 'float', 'format', 'frozenset', 'getattr', 
    'globals', 'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'intern', 'isinstance', 
    'issubclass', 'iter', 'len', 'license', 'list', 'locals', 'long', 'map', 'max', 'min', 'next', 
    'object', 'oct', 'open', 'ord', 'pow', 'print', 'property', 'quit', 'range', 'raw_input', 'reduce', 'reload', 
    'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 
    'sum', 'super', 'tuple', 'type', 'unichr', 'unicode', 'vars', 'xrange', 'zip') 
    

    def __init__(self, edit):
        self.textedit = edit
        document = edit.document()
        QSyntaxHighlighter.__init__(self, document)

        base_format = QTextCharFormat()
        base_format.setFont(edit.font())
        self.base_format = base_format
        self.document = document
        
        self.updateHighlighter(base_format.font())


    def highlightBlock(self, text):
        self.setCurrentBlockState(0)
        
        if text.trimmed().isEmpty():
            self.setFormat(0, len(text), self.empty_format)
            return
        
        self.setFormat(0, len(text), self.base_format)
        
        startIndex = 0
        if self.previousBlockState() != 1:
            startIndex = self.multiLineStringBegin.indexIn(text)
        
        if startIndex > -1:
            self.highlightRules(text, 0, startIndex)
        else:
            self.highlightRules(text, 0, len(text))
        
        while startIndex >= 0:
            endIndex = self.multiLineStringEnd.indexIn(text, 
                  startIndex + len(self.multiLineStringBegin.pattern()))
            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLength = text.length() - startIndex
            else:
                commentLength = endIndex - startIndex + \
                                self.multiLineStringEnd.matchedLength()
                self.highlightRules(text, endIndex, len(text))
            
            self.setFormat(startIndex, commentLength, self.multiLineStringFormat)
            startIndex = self.multiLineStringBegin.indexIn(text, 
                                           startIndex + commentLength)
    
    
    def highlightRules(self, text, start, finish):
        for expression, format in self.rules:
            index = expression.indexIn(text, start)
            while index >= start and index < finish:
                length = expression.matchedLength()
                self.setFormat(index, min(length, finish - index), format)
                index = expression.indexIn(text, index + length)
    

    def updateFonts(self, font):    
        self.base_format.setFont(font)
        self.empty_format = QTextCharFormat(self.base_format)
        #self.empty_format.setFontPointSize(font.pointSize()/4.0)
        self.keywordFormat = QTextCharFormat(self.base_format)
        self.keywordFormat.setFontWeight(QtGui.QFont.Bold)
        self.keywordFormat.setForeground(QtCore.Qt.black)
        #self.keywordFormat.setFontWeight(QFont.Bold)
        self.builtinFormat = QTextCharFormat(self.base_format)
        self.builtinFormat.setForeground(QtCore.Qt.blue)
        self.magicFormat = QTextCharFormat(self.base_format)
        self.magicFormat.setForeground(QtCore.Qt.red)
        #self.qtFormat = QTextCharFormat(self.base_format)
        #self.qtFormat.setForeground(QColor(scheme.syntax_qt))
        ##self.qtFormat.setFontWeight(QFont.Bold)
        self.selfFormat = QTextCharFormat(self.base_format)
        self.selfFormat.setForeground(QtCore.Qt.darkGray)
        #self.selfFormat.setFontItalic(True)
        self.singleLineCommentFormat = QTextCharFormat(self.base_format)
        self.singleLineCommentFormat.setForeground(QtCore.Qt.gray)
        self.singleLineCommentFormat.setFontItalic(True)
        self.multiLineStringFormat = QTextCharFormat(self.base_format)
        self.multiLineStringFormat.setForeground(QtCore.Qt.green)
        #self.multiLineStringFormat.setBackground(QBrush(Qt.green))
        self.quotationFormat1 = QTextCharFormat(self.base_format)
        self.quotationFormat1.setForeground(QtCore.Qt.darkMagenta)
        self.quotationFormat2 = QTextCharFormat(self.base_format)
        self.quotationFormat2.setForeground(QtCore.Qt.darkBlue)
        self.numFormat = QTextCharFormat(self.base_format)
        self.numFormat.setForeground(QtCore.Qt.darkCyan)


    def updateRules(self):
        self.rules = []
        self.rules += map(lambda s: (QRegExp(r"\b"+s+r"\b"),
                          self.keywordFormat), self.keywords)
        self.rules += map(lambda s: (QRegExp(r"\b"+s+r"\b"),
                          self.builtinFormat), self.builtins)

        self.rules.append((QRegExp(r"\b__[a-z]+__\b"), self.magicFormat))
        self.rules.append((QRegExp(r"\bself\b"), self.selfFormat))
        self.rules.append((QRegExp(r"\b\d+(\.\d*)?\b"), self.numFormat))
        #self.rules.append((QRegExp(r"\bQ([A-Z][a-z]*)+\b"), self.qtFormat))
        self.rules.append((QRegExp(r"#[^\n]*"), self.singleLineCommentFormat))
        self.multiLineStringBegin = QRegExp(r'\"\"\"')
        self.multiLineStringEnd = QRegExp(r'\"\"\"')
        self.rules.append((QRegExp(r'\"[^\n\"]*\"'), self.quotationFormat1))
        self.rules.append((QRegExp(r"'[^\n\']*'"), self.quotationFormat2))
    

    def updateHighlighter(self, font):    
        self.updateFonts(font)
        self.updateRules()
        self.setDocument(self.document)