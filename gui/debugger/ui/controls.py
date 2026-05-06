from ...qt.QtWidgets import *
from ...qt import QtSignal
from ...qt.QtGui import QIcon

class DebugControls(QWidget):
    continue_clicked = QtSignal()
    step_line_clicked = QtSignal()
    step_into_clicked = QtSignal()
    step_out_clicked = QtSignal()
    stop_clicked = QtSignal()

    def __init__(self):
        super().__init__()
        button_layout = QHBoxLayout(self)
        style = self.style()

        self.continue_button = QPushButton()
        self.continue_button.setIcon(QIcon("gui/debugger/ui/icons/play.svg"))
        self.continue_button.clicked.connect(self.continue_clicked)
        self.continue_button.setEnabled(False)
        self.continue_button.setToolTip("Continue execution until the next breakpoint.")

        self.step_line_button = QPushButton()
        self.step_line_button.setIcon(QIcon("gui/debugger/ui/icons/step.svg"))
        self.step_line_button.clicked.connect(self.step_line_clicked)
        self.step_line_button.setEnabled(False)
        self.step_line_button.setToolTip("Execute the next line of code.")

        self.step_into_button = QPushButton()
        self.step_into_button.setIcon(QIcon("gui/debugger/ui/icons/step_in.svg"))
        self.step_into_button.clicked.connect(self.step_into_clicked)
        self.step_into_button.setEnabled(False)
        self.step_into_button.setToolTip("Step into the next function call.")

        self.step_out_button = QPushButton()
        self.step_out_button.setIcon(QIcon("gui/debugger/ui/icons/step_out.svg"))
        self.step_out_button.clicked.connect(self.step_out_clicked)
        self.step_out_button.setEnabled(False)
        self.step_out_button.setToolTip("Step out of the current function.")

        self.stop_button = QPushButton()
        self.stop_button.setIcon(style.standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_clicked)
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("Stop the debugger and disconnect from the program.")

        # Add buttons to layout
        for btn in [
            self.continue_button,
            self.step_line_button,
            self.step_into_button,
            self.step_out_button,
            self.stop_button
        ]:
            button_layout.addWidget(btn)
