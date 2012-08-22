/****************************************************************************
**
** Copyright (C) 2011 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Nokia Corporation and its Subsidiary(-ies) nor
**     the names of its contributors may be used to endorse or promote
**     products derived from this software without specific prior written
**     permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
** $QT_END_LICENSE$
**
****************************************************************************/

#ifndef PLASK_GUI_MAINWINDOW_H
#define PLASK_GUI_MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QTreeView>
#include <QListView>
#include <QtTreePropertyBrowser>
#include <QItemSelection>


#include "document.h"
#include "view/elementview.h"
#include "creatorslist.h"

QT_BEGIN_NAMESPACE
class QAction;
class QListWidget;
class QMenu;
class QTextEdit;
class QTreeView;
class QListView;
class QtTreePropertyBrowser;
class QItemSelection;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();

private slots:
    void newDocument();
    void open();
    void save();
    void print();
    void undo();
    void editSelected();
    void about();
    void treeSelectionChanged(const QItemSelection & selected, const QItemSelection & deselected);
    void treeRemoveSelected();
    void treeAddCartesian2d();
    void treeAddCartesian3d();
    void treeAddCylindric();
    void treeAddBlock2D();

private:

    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void createDockWindows();

    ElementViewer *view;    //TODO should accept only drop with source() != 0 (from this application)
    QTreeView *treeView;    //TODO should accept only drop with source() != 0 (from this application)
    QtTreePropertyBrowser *propertyTree;
    QListView *creatorsList;

    Document document;
    CreatorsListModel creators;

    QMenu *fileMenu;
    QMenu *editMenu;
    QMenu *geometryMenu;
    QMenu *viewMenu;
    QMenu *helpMenu;
    QToolBar *fileToolBar;
    QToolBar *editToolBar;
    QToolBar *viewToolBar;
    QAction *newDocumentAct;
    QAction *openAct;
    QAction *saveAct;
    QAction *printAct;
    QAction *undoAct;
    QAction *editSelectedAct;
    QAction *aboutAct;
    QAction *aboutQtAct;
    QAction *quitAct;

    QAction *treeRemoveAct;
    QAction *treeAddCartesian2dAct;
    QAction *treeAddCylindricAct;
    QAction *treeAddCartesian3dAct;
    QAction *treeAddBlockAct;


    QAction *zoomInAct;
    QAction *zoomOutAct;
    

};

#endif  // PLASK_GUI_MAINWINDOW_H
