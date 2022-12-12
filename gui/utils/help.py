# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

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
import os
import os.path
import webbrowser

from ..qt.QtCore import *
from ..qt.QtGui import *
from ..qt.QtWidgets import *
from ..qt.QtHelp import *

from .config import CONFIG
from .widgets import set_icon_size

HELP_URL = 'https://docs.plask.app'

HELP_DIR = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'doc', 'plask')
HELP_DIR = os.environ.get('PLASK_HELP_DIR', HELP_DIR)

HELP_FILE = os.path.join(HELP_DIR, 'plask.qch')
COLLECTION_FILE = os.path.join(HELP_DIR, 'plask.qhc')

HELP_WINDOW = None
HELP_ENGINE = None


class HelpBrowser(QTextBrowser):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self.open_link)
        font = self.font()
        font.setPointSize(int(CONFIG['help/fontsize']))
        self.setFont(font)

    def loadResource(self, typ, url):
        if typ < 4 and HELP_ENGINE:
            if url.isRelative():
                url = self.source().resolved(url)
            return HELP_ENGINE.fileData(url)

    def open_link(self, url):
        if url.isRelative():
            url = self.source().resolved(url)
        if url.scheme() == 'qthelp':
            self.setSource(url)
        else:
            webbrowser.open(url.toString())


class HelpWindow(QSplitter):

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PLaSK Help")
        self.setWindowIcon(QIcon.fromTheme('help-contents'))

        try:
            main_window.config_changed.connect(self.reconfig)
        except AttributeError:
            pass

        browser_area = QWidget()
        browser_layout = QVBoxLayout()
        self.toolbar = QToolBar()
        set_icon_size(self.toolbar)
        self.browser = HelpBrowser()
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

        self.setOrientation(Qt.Orientation.Horizontal)
        tabs = QTabWidget(self)
        tabs.setMaximumWidth(480)
        tabs.addTab(HELP_ENGINE.contentWidget(), "Contents")
        tabs.addTab(HELP_ENGINE.indexWidget(), "Index")
        self.addWidget(tabs)
        self.addWidget(browser_area)

        HELP_ENGINE.contentWidget().linkActivated.connect(self.browser.setSource)
        HELP_ENGINE.indexWidget().linkActivated.connect(self.browser.setSource)
        self.browser.sourceChanged.connect(self.update_content_widget)
        HELP_ENGINE.contentModel().contentsCreated.connect(
            lambda: self.update_content_widget(self.browser.source()))

        self.namespace = HELP_ENGINE.namespaceName(HELP_FILE)
        self.prefix = 'qthelp://{}/doc/'.format(self.namespace)
        self.browser.setSource(QUrl(self.prefix+'index.html'))

        self.resize(int(0.85 * main_window.width()), int(0.85 * main_window.height()))

    def reconfig(self):
        font = self.browser.font()
        font.setPointSize(int(CONFIG['help/fontsize']))
        self.browser.setFont(font)

    def update_content_widget(self, url):
        content_widget = HELP_ENGINE.contentWidget()
        return content_widget.setCurrentIndex(content_widget.indexOf(url))

    def update_toolbar(self):
        self.prev.setEnabled(self.browser.isBackwardAvailable())
        self.next.setEnabled(self.browser.isForwardAvailable())

    def show_help_for_keyword(self, keyword):
        if HELP_ENGINE:
            links = HELP_ENGINE.linksForIdentifier(keyword)
            if links.count():
                self.browser.setSource(links.constBegin().value())

    def show_page(self, page):
        self.browser.setSource(QUrl('{}{}.html'.format(self.prefix, page)))

    def open_browser(self):
        page = self.browser.source().path()[4:-5]
        if page[-5:] == 'index': page = page[:-5]
        if page:
            if not page.startswith('/'):
                webbrowser.open("{}/{}".format(HELP_URL, page))
            else:
                webbrowser.open("{}{}".format(HELP_URL, page))
        else:
            webbrowser.open("{}".format(HELP_URL))


def init_help_engine(parent=None):
    global HELP_ENGINE
    if HELP_ENGINE is None:
        HELP_ENGINE = QHelpEngine(COLLECTION_FILE, parent)
        if not HELP_ENGINE.setupData():
            HELP_ENGINE = None
            return False
    return True


def open_help(page=None, main_window=None):
    if os.path.exists(COLLECTION_FILE) and not CONFIG['help/online'] and init_help_engine(main_window):
        global HELP_WINDOW
        if HELP_WINDOW is None:
            HELP_WINDOW = HelpWindow(main_window)
        else:
            HELP_WINDOW.raise_()
        if page:
            HELP_WINDOW.show_page(page)
        HELP_WINDOW.show()
    elif page:
        if not page.startswith('/'):
            webbrowser.open("{}/{}".format(HELP_URL, page))
        else:
            webbrowser.open("{}{}".format(HELP_URL, page))
    else:
        webbrowser.open(HELP_URL)
