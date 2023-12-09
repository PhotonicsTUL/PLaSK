from . import QT_API

if QT_API == 'PyQt6':
    from PyQt6.QtCore import *
elif QT_API == 'PySide6':
    from PySide6.QtCore import *
elif QT_API == 'PyQt5':
    from PyQt5.QtCore import *
else:  # QT_API == 'PySide2':
    from PySide2.QtCore import *
