# coding: utf8
# Copyright (C) 2014 Photonics Group, Lodz University of Technology
#
# This program is free software you can redistribute it and/or modify it
# under the terms of GNU General Public License as published by the
# Free Software Foundation either version 2 of the license, or (at your
# opinion) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import sys
import os.path
import webbrowser

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..qt.QtHelp import *

HELP_URL = 'http://fizyka.p.lodz.pl/en/plask-user-guide/'

HELP_DIR = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'doc', 'plask')
HELP_FILE = os.path.join(HELP_DIR, 'plask.qch')
COLLECTION_FILE = os.path.join(HELP_DIR, 'plask.qhc')

HELP_WINDOW = None

class HelpBrowser(QTextBrowser):

    def __init__(self, help_engine, parent=None):
        super(HelpBrowser, self).__init__(parent)
        self.help_engine = help_engine

    def loadResource(self, typ, url):
        if typ < 4 and self.help_engine:
            if url.isRelative():
                url = self.source().resolved(url)
            return self.help_engine.fileData(url)


class HelpWindow(QSplitter):

    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.setWindowTitle("PLaSK Help")
        self.setWindowIcon(QIcon.fromTheme('help-contents'))

        self.help_engine = QHelpEngine(COLLECTION_FILE, self)

        if not self.help_engine.setupData():
            self.help_engine = None
            label = QLabel("ERROR: Could not load help file!")
            label.setAlignment(Qt.AlignCenter)
            self.resize(600, 400)
            self.addWidget(label)
            return

        browser_area = QWidget()
        browser_layout = QVBoxLayout()
        self.toolbar = QToolBar()
        self.browser = HelpBrowser(self.help_engine)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.addWidget(self.toolbar)
        browser_layout.addWidget(self.browser)
        browser_area.setLayout(browser_layout)

        home = QAction(QIcon.fromTheme('go-home'), "Home", self.toolbar)
        home.triggered.connect(self.browser.home)
        self.prev = QAction(QIcon.fromTheme('go-previous'), "Back", self.toolbar)
        self.prev.triggered.connect(self.browser.backward)
        self.prev.setEnabled(False)
        self.next = QAction(QIcon.fromTheme('go-next'), "Forward", self.toolbar)
        self.next.triggered.connect(self.browser.forward)
        self.next.setEnabled(False)
        self.toolbar.addActions([home, self.prev, self.next])
        self.browser.historyChanged.connect(self.update_toolbar)

        web = QAction(QIcon.fromTheme('globe'), "Open in Browser", self.toolbar)
        web.triggered.connect(self.open_browser)
        self.toolbar.addSeparator()
        self.toolbar.addAction(web)

        self.setOrientation(Qt.Horizontal)
        tabs = QTabWidget(self)
        tabs.setMaximumWidth(480)
        tabs.addTab(self.help_engine.contentWidget(), "Contents")
        tabs.addTab(self.help_engine.indexWidget(), "Index")
        self.addWidget(tabs)
        self.addWidget(browser_area)

        self.help_engine.contentWidget().linkActivated.connect(self.browser.setSource)
        self.help_engine.indexWidget().linkActivated.connect(self.browser.setSource)
        self.browser.sourceChanged.connect(self.update_content_widget)
        self.help_engine.contentModel().contentsCreated.connect(
            lambda: self.update_content_widget(self.browser.source()))

        self.namespace = self.help_engine.namespaceName(HELP_FILE)
        self.prefix = 'qthelp://{}/doc/'.format(self.namespace)
        self.browser.setSource(QUrl(self.prefix+'index.html'))

        self.resize(1200, 800)

    def update_content_widget(self, url):
        content_widget = self.help_engine.contentWidget()
        return content_widget.setCurrentIndex(content_widget.indexOf(url))

    def update_toolbar(self):
        self.prev.setEnabled(self.browser.isBackwardAvailable())
        self.next.setEnabled(self.browser.isForwardAvailable())

    def show_help_for_keyword(self, keyword):
        if self.help_engine:
            links = self.help_engine.linksForIdentifier(keyword)
            if links.count():
                self.browser.setSource(links.constBegin().value())

    def show_page(self, page):
        self.browser.setSource(QUrl('{}{}.html'.format(self.prefix, page)))

    def open_browser(self):
        page = self.browser.source().path()[4:-5]
        if page[-5:] == 'index': page = page[:-5]
        if page:
            webbrowser.open("{}{}/".format(HELP_URL, page))
        else:
            webbrowser.open("{}".format(HELP_URL))


def open_help(page=None):
    if os.path.exists(COLLECTION_FILE):
        global HELP_WINDOW
        if HELP_WINDOW is None:
            HELP_WINDOW = HelpWindow()
        else:
            HELP_WINDOW.raise_()
        if page:
            HELP_WINDOW.show_page(page)
        HELP_WINDOW.show()
    elif page:
        webbrowser.open("{}{}/".format(HELP_URL, page))
    else:
        webbrowser.open(HELP_URL)
