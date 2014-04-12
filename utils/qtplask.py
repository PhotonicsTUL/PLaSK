#!/usr/bin/python
#coding: utf8

import sys
import os
import base64
import subprocess
import ctypes
from time import localtime, strftime, sleep

sys.argv[0] = os.path.abspath(sys.argv[0])

from PySide import QtCore, QtGui

APP_ICON = '''
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
yU/85en/BUFo1JV1G1EMAAAAAElFTkSuQmCC
'''

STOP_ICON = '''
iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAABF1BMVEX///+AAACoAACtAACHAACK
AACRAACYAACcAACqAACAAACoAACvBAScAACHDg6OAABuAABtAABqAACiAACWKyt/HhaCJBvSeHXD
SknJYF2dHxy/NCrFRz+kHxOkHxWkIRqkIx6lHhKmHhCrJBevJxuzMSazKx++Sj/DSkDITEPKVkyA
IyGBIyGhPjSyQzpqIx9qIyB7PDJZIRxbIRu9AAC+AAC/OTPBEQrCMzPCTUjDOjrFOTLGEgzHEgzJ
Pj7LRD/LTkrMRkbQAADQEgzTVlbTV1TTZWDUYl7WYF3XY2HbExDbFBHfEw/fFBLfaGHjAADlLyrl
MCzlMi3lMi7lc3HmbW3mdnXpjYnvFBLvFRPviYfwjYr1AABjjkEhAAAANHRSTlMAMjIySEhISEhI
SUlJSktPUFFS0Nfd3uTm6vL09fr6+vr6+vr6+vv7+/v7/Pz8/P39/f7+1W7eYgAAAKNJREFUeNqN
jMUWglAABR8ggoqKYGJidyt2AHZgYfL/3+HhcVjrbO6ZWVzwF1iI0ocKYIaj4XfRBoCj8ImgMLBH
9Z53uqs39cHCQGYX8qbSWMnrHAkgRHo6mfXnywxhvlpTw96gm7QCE5wX2h2Bx023RCVxNBalmMVw
e3yvbFvNnXJIuGDwnbRnjWbqL+3shQHhLiUaAKZ85RBgFL9HHyYI/SdfQ7MTa6WMvbgAAAAASUVO
RK5CYII=
'''

class MainWindow(QtGui.QMainWindow):
    '''Main Qt window class'''

    def setupUi(self, config):
        self.setObjectName("PLaSK")
        self.setWindowTitle(self.tr("PLaSK"))

        space = QtGui.QDesktopWidget().availableGeometry(self)
        self.resize(config.value("window/size", QtCore.QSize(int(0.8*space.width()), int(0.9*space.height()))))
        qr = self.frameGeometry()
        qr.moveCenter(QtGui.QDesktopWidget().screenGeometry(self).center())
        self.move(config.value("window/pos", QtCore.QPoint(qr.topLeft())))

        icon_pixmap = QtGui.QPixmap()
        icon_pixmap.loadFromData(QtCore.QByteArray.fromBase64(APP_ICON))
        icon = QtGui.QIcon()
        icon.addPixmap(icon_pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)

        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setEnabled(True)

        layout = QtGui.QVBoxLayout()

        self.tabBar = QtGui.QTabBar()
        layout.addWidget(self.tabBar)

        #splitter =  QtGui.QSplitter()
        #splitter.setOrientation(QtCore.Qt.Vertical)
        #layout.addWidget(splitter)

        font = QtGui.QFont()
        if sys.platform == 'win32':
            font.setFamily("Consolas")
        elif sys.platform == 'darwin':
            font.setFamily("Monaco")
        else:
            font.setFamily("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setPointSize(10)
        #self.outputView = QtGui.QTextEdit()
        #self.outputView.setReadOnly(True)
        #self.outputView.setAcceptRichText(False)
        #self.outputView.setFont(font)
        #splitter.addWidget(self.outputView)
        self.messagesView = QtGui.QTextEdit()
        self.messagesView.setReadOnly(True)
        self.messagesView.setAcceptRichText(True)
        self.messagesView.setFont(font)
        self.messagesView.append(self.tr("Press F5 to start computations..."))
        #splitter.addWidget(self.messagesView)
        layout.addWidget(self.messagesView)

        self.centralwidget.setLayout(layout)
        self.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setTitle(self.tr("&File"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setTitle(self.tr("&View"))
        self.actionRun = QtGui.QAction(self)
        self.actionRun.setText(self.tr("&Run Computations..."))
        self.actionRun.setShortcut("F5")
        self.actionQuit = QtGui.QAction(self)
        self.actionQuit.setText(self.tr("&Quit"))
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionError = QtGui.QAction(self)
        self.actionError.setText(self.tr("&Error"))
        self.actionError.setCheckable(True)
        self.actionError.setChecked(config.value('view/error', 'true')=='true')
        self.actionWarning = QtGui.QAction(self)
        self.actionWarning.setText(self.tr("&Warning"))
        self.actionWarning.setCheckable(True)
        self.actionWarning.setChecked(config.value('view/warning', 'true')=='true')
        self.actionInfo = QtGui.QAction(self)
        self.actionInfo.setText(self.tr("&Info"))
        self.actionInfo.setCheckable(True)
        self.actionInfo.setChecked(config.value('view/info', 'true')=='true')
        self.actionResult = QtGui.QAction(self)
        self.actionResult.setText(self.tr("&Result"))
        self.actionResult.setCheckable(True)
        self.actionResult.setChecked(config.value('view/result', 'true')=='true')
        self.actionData = QtGui.QAction(self)
        self.actionData.setText(self.tr("&Data"))
        self.actionData.setCheckable(True)
        self.actionData.setChecked(config.value('view/data', 'true')=='true')
        self.actionDetail = QtGui.QAction(self)
        self.actionDetail.setText(self.tr("De&tail"))
        self.actionDetail.setCheckable(True)
        self.actionDetail.setChecked(config.value('view/detail', 'true')=='true')
        self.actionDebug = QtGui.QAction(self)
        self.actionDebug.setText(self.tr("De&bug"))
        self.actionDebug.setCheckable(True)
        self.actionDebug.setChecked(config.value('view/debug', 'false')=='true')
        self.menuFile.addAction(self.actionRun)
        self.menuFile.addAction(self.actionQuit)
        self.menuView.addAction(self.actionError)
        self.menuView.addAction(self.actionWarning)
        self.menuView.addAction(self.actionInfo)
        self.menuView.addAction(self.actionResult)
        self.menuView.addAction(self.actionData)
        self.menuView.addAction(self.actionDetail)
        self.menuView.addAction(self.actionDebug)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        dan2xpl = self.find_tool('dan2xpl')
        xpl2dan = self.find_tool('xpl2dan')
        if dan2xpl or xpl2dan or winsparkle:
            self.menuTools = QtGui.QMenu(self.menubar)
            self.menuTools.setTitle(self.tr("&Tools"))
            if dan2xpl:
                self.actionDanXpl = QtGui.QAction(self)
                self.actionDanXpl.setText(self.tr("Convert DAN to &XPL..."))
                self.actionDanXpl.triggered.connect(lambda: self.runConvert(dan2xpl, 'RPSMES files (*.dan)'))
                self.menuTools.addAction(self.actionDanXpl)
            if xpl2dan:
                self.actionXplDan = QtGui.QAction(self)
                self.actionXplDan.setText(self.tr("Convert XPL to &DAN..."))
                self.actionXplDan.triggered.connect(lambda: self.runConvert(xpl2dan, 'XPL files (*.xpl)'))
                self.menuTools.addAction(self.actionXplDan)
            if winsparkle:
                if dan2xpl or xpl2dan: self.menuTools.addSeparator()
                try:
                    self.actionWinSparkleAutoupdate = QtGui.QAction(self)
                    self.actionWinSparkleAutoupdate.setText(self.tr("Automatic Updates"))
                    self.actionWinSparkleAutoupdate.setCheckable(True)
                    self.actionWinSparkleAutoupdate.setChecked(winsparkle.win_sparkle_get_automatic_check_for_updates())
                    self.actionWinSparkleAutoupdate.triggered.connect(
                        lambda: winsparkle.win_sparkle_set_automatic_check_for_updates(int(self.actionWinSparkleAutoupdate.isChecked()))
                    )
                    self.menuTools.addAction(self.actionWinSparkleAutoupdate)
                except AttributeError:
                    pass
                self.actionWinSparkle = QtGui.QAction(self)
                self.actionWinSparkle.setText(self.tr("Check for Updates Now..."))
                self.actionWinSparkle.triggered.connect(lambda: winsparkle.win_sparkle_check_update_with_ui())
                self.menuTools.addAction(self.actionWinSparkle)
            self.menubar.addAction(self.menuTools.menuAction())

        self.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(self)
        self.setStatusBar(self.statusbar)

        stop_pixmap = QtGui.QPixmap()
        stop_pixmap.loadFromData(QtCore.QByteArray.fromBase64(STOP_ICON))
        self.stopicon = QtGui.QIcon()
        self.stopicon.addPixmap(stop_pixmap, QtGui.QIcon.Normal, QtGui.QIcon.Off)


    def dragEnterEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if (urls and urls[0].scheme() == 'file'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        data = event.mimeData()
        urls = data.urls()
        if (urls and urls[0].scheme() == 'file'):
            filepath = str(urls[0].path())
            if os.name == 'nt':
                filepath = filepath[1:]
            self.start_plask(filepath)


    def __init__(self):
        #self.outputs = []
        self.messages = []
        self.threads = []

        config = QtCore.QSettings("PLaSK", "qtplask")

        try:
            self.last_dir = os.environ['HOME']
        except:
            self.last_dir = ""
        self.last_dir = config.value('recentdir', self.last_dir)

        super(MainWindow, self).__init__()

        self.setupUi(config)

        self.setAcceptDrops(True)

        self.actionQuit.triggered.connect(QtCore.QCoreApplication.instance().quit)
        self.actionRun.triggered.connect(self.runFile)
        self.tabBar.currentChanged.connect(self.switch_tab)

        self.actionError.triggered.connect(self.switch_tab)
        self.actionWarning.triggered.connect(self.switch_tab)
        self.actionInfo.triggered.connect(self.switch_tab)
        self.actionResult.triggered.connect(self.switch_tab)
        self.actionData.triggered.connect(self.switch_tab)
        self.actionDetail.triggered.connect(self.switch_tab)
        self.actionDebug.triggered.connect(self.switch_tab)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_outputs)
        self.timer.start(250)


    def find_tool(self, name):
        fname = os.path.join(os.path.dirname(sys.argv[0]), name)
        if os.path.exists(fname+'.py'):
            return fname+'.py'
        elif os.path.exists(fname):
            return fname
        else:
            return None


    def runConvert(self, tool, filter):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, self.tr("Choose file to convert"), self.last_dir, self.tr(filter))
        if not fname: return
        self.last_dir = os.path.dirname(fname)
        self.start_plask(tool, fname)


    def runFile(self):
        '''Load and run XPL script in an external program'''
        fname, _ = QtGui.QFileDialog.getOpenFileName(self, self.tr("Choose file to run"), self.last_dir, self.tr("PLaSK files (*.xpl *.py)"))
        if not fname: return
        self.last_dir = os.path.dirname(fname)
        self.start_plask(fname)


    def start_plask(self, fname, *args):
        #self.outputs.append([])
        self.messages.append([])
        idx = len(self.messages)-1
        self.tabBar.addTab("%s @ %s" % (os.path.basename(fname), strftime('%X')))
        button = QtGui.QPushButton(self)
        button.setFlat(True)
        button.setFixedSize(16, 14)
        button.setIcon(self.stopicon)
        button.setToolTip(self.tr("Abort computations"))
        self.tabBar.setTabButton(idx, QtGui.QTabBar.RightSide, button)
        self.tabBar.setCurrentIndex(idx)

        thread = PlaskThread(fname, self.last_dir, self.messages[-1], *args)
        thread.finished.connect(lambda: self.set_finished(idx))
        self.threads.append(thread)
        button.clicked.connect(thread.kill_process)
        thread.start()


    def set_finished(self, idx):
        self.tabBar.setTabButton(idx, QtGui.QTabBar.RightSide, None)
        self.tabBar.setTabText(idx, self.tabBar.tabText(idx) + " (%s)" % strftime('%X'))


    def switch_tab(self):
        self.messagesView.clear()
        self.printed_lines = 0
        self.update_outputs()


    def update_outputs(self):
        n = self.tabBar.currentIndex()
        if n == -1: return
        move = self.messagesView.verticalScrollBar().value() == self.messagesView.verticalScrollBar().maximum()
        total_lines = len(self.messages[n])
        lines = []
        if self.printed_lines != total_lines:
            for line in self.messages[n][self.printed_lines:total_lines]:
                cat = line[19:26].rstrip()
                if cat in ('red', '#800000') and not self.actionError.isChecked(): continue
                if cat == 'brown' and not self.actionWarning.isChecked(): continue
                if cat == 'blue' and not self.actionInfo.isChecked(): continue
                if cat == 'green' and not self.actionResult.isChecked(): continue
                if cat == '#006060' and not self.actionData.isChecked(): continue
                if cat == 'black' and not self.actionDetail.isChecked(): continue
                if cat == 'gray' and not self.actionDebug.isChecked(): continue
                lines.append(line)
            if lines: self.messagesView.append("<br/>\n".join(lines))
            self.printed_lines = total_lines
            if move: self.messagesView.moveCursor(QtGui.QTextCursor.End)


    def quitting(self):
        config = QtCore.QSettings("PLaSK", "qtplask")
        config.setValue("recentdir", self.last_dir)
        config.setValue("window/size", self.size())
        config.setValue("window/pos", self.pos())
        config.setValue('view/error', self.actionError.isChecked())
        config.setValue('view/warning', self.actionWarning.isChecked())
        config.setValue('view/info', self.actionInfo.isChecked())
        config.setValue('view/result', self.actionResult.isChecked())
        config.setValue('view/data', self.actionData.isChecked())
        config.setValue('view/detail', self.actionDetail.isChecked())
        config.setValue('view/debug', self.actionDebug.isChecked())

        for thread in self.threads:
            if thread.isRunning():
                thread.kill_process()
                thread.terminate()

        sleep(1)


class PlaskThread(QtCore.QThread):

    def __init__(self, fname, dirname, lines, *args):
        super(PlaskThread, self).__init__()

        try:
            si = subprocess.STARTUPINFO()
            si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
        except AttributeError:
            self.proc = subprocess.Popen(['plask', '-ldebug', '-u', fname] + list(args),
                                         cwd=dirname, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:
            self.proc = subprocess.Popen(['plask', '-ldebug', '-u', '-w', fname] + list(args), startupinfo=si,
                                         cwd=dirname, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        self.lines = lines
        self.terminated.connect(self.kill_process)

    def run(self):
        while self.proc.poll() is None:
            line = self.proc.stdout.readline().rstrip()
            if not line: continue
            cat = line[:15]
            line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if   cat == "CRITICAL ERROR:": color = "red    "
            elif cat == "ERROR         :": color = "red    "
            elif cat == "WARNING       :": color = "brown  "
            elif cat == "INFO          :": color = "blue   "
            elif cat == "RESULT        :": color = "green  "
            elif cat == "DATA          :": color = "#006060"
            elif cat == "DETAIL        :": color = "black  "
            elif cat == "ERROR DETAIL  :": color = "#800000"
            elif cat == "DEBUG         :": color = "gray   "
            else: color = "black; font-weight:bold"
            line = line.replace(' ', '&nbsp;')
            self.lines.append('<span style="color:%s;">%s</span>' % (color, line))

    def kill_process(self):
        self.proc.terminate()

if __name__ == "__main__":

    try:
        winsparkle = ctypes.CDLL('WinSparkle.dll')
    except OSError:
        winsparkle = None
    else:
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESTDHANDLES | subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        proc = subprocess.Popen(['plask', '-V'], startupinfo=si, stdout=subprocess.PIPE)
        version, err = proc.communicate()
        prog, ver = version.strip().split()
        wp = ctypes.c_wchar_p
        winsparkle.win_sparkle_set_app_details(wp("PLaSK"), wp("PLaSK"), wp(ver))
        winsparkle.win_sparkle_set_appcast_url("http://phys.p.lodz.pl/appcast/plask.xml")
        winsparkle.win_sparkle_set_registry_path("Software\\PLaSK\\plask\\WinSparkle")
        winsparkle.win_sparkle_init()

    try:
        fname = sys.argv[1]
    except IndexError:
        fname = None
    else:
        fname = os.path.realpath(fname)
    app = QtGui.QApplication(sys.argv)
    mainwindow = MainWindow()
    app.aboutToQuit.connect(mainwindow.quitting)
    mainwindow.show()
    if fname:
        mainwindow.start_plask(fname)
    exit_code = app.exec_()
    if winsparkle:
        winsparkle.win_sparkle_cleanup()
    sys.exit(exit_code)
