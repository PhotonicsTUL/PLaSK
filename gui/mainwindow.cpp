#include <QtGui>
#include <QFileDialog>
#include <QApplication>
#include <QStatusBar>
#include <QMessageBox>
#include <QAction>
#include <QMenuBar>
#include <QMenu>
#include <QToolBar>
#include <QDockWidget>

#include "modelext/draw.h"

#include "mainwindow.h"

#include "utils/draw.h"
#include <plask/geometry/leaf.h>
#include <plask/geometry/space.h>

MainWindow::MainWindow()
    : propertyTree(new QtTreePropertyBrowser(this)), document(*propertyTree)
{
    view = new ObjectViewer(this);
    view->setModel(&document.treeModel);

    setCentralWidget(view);

    createActions();
    createMenus();
    createToolBars();
    createStatusBar();
    createDockWindows();

    view->setSelectionModel(treeView->selectionModel());
    //treeView->setSelectionModel(selectionModel);

    setWindowTitle(tr("PLaSK GUI"));

    setUnifiedTitleAndToolBarOnMac(true);
}

void MainWindow::newDocument() {
    document.clear();
}

void MainWindow::print()
{
#ifndef QT_NO_PRINTDIALOG
    /*QTextDocument *document = textEdit->document();
    QPrinter printer;

    QPrintDialog *dlg = new QPrintDialog(&printer, this);
    if (dlg->exec() != QDialog::Accepted)
        return;

    document->print(&printer);

    statusBar()->showMessage(tr("Ready"), 2000);*/
#endif
}

void MainWindow::open() {
    QString loadName = QFileDialog::getOpenFileName(this, tr("Choose name of experiment file to open"), ".", tr("XPL (*.xpl)"));
    if (loadName.isEmpty()) return;
    document.open(loadName);
    view->setRootIndex(document.treeModel.index(0, 0));

    /*view->setTransform(flipVertical);
    view->scale(10.0, 10.0);
    scene->addItem(new GeometryObjectItem(document.manager.getRootObject<plask::GeometryObjectD<2>>(0)));
    scene->addLine(-10.0, 0.0, 10.0, 0.0, QPen(Qt::red));
    scene->addLine(0.0, -10.0, 0.0, 10.0, QPen(Qt::red));*/
}


void MainWindow::save()
{
    QString fileName = QFileDialog::getSaveFileName(this,
                        tr("Choose a file name"), ".",
                        tr("XPL (*.xpl)"));
    if (fileName.isEmpty())
        return;

    QApplication::setOverrideCursor(Qt::WaitCursor);
    try {
        document.save(fileName.toStdString());
        statusBar()->showMessage(tr("Saved '%1'").arg(fileName), 2000);
    } catch (std::exception& exp) {
        QMessageBox::warning(this, tr("PLaSK GUI"),
                             tr("Cannot write file %1:\n%2")
                             .arg(fileName).arg(exp.what()));
    }
    QApplication::restoreOverrideCursor();
    
   /* QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        QMessageBox::warning(this, tr("PLaSK GUI"),
                             tr("Cannot write file %1:\n%2.")
                             .arg(fileName)
                             .arg(file.errorString()));
        return;
    }

    QTextStream out(&file);
    QApplication::setOverrideCursor(Qt::WaitCursor);
    //out << textEdit->toHtml();
    QApplication::restoreOverrideCursor();

    statusBar()->showMessage(tr("Saved '%1'").arg(fileName), 2000);*/
}

void MainWindow::undo()
{
    /*QTextDocument *document = textEdit->document();
    document->undo();*/
}

void MainWindow::editSelected() {
    auto selIndexes = treeView->selectionModel()->selectedIndexes();
    if (!selIndexes.empty())
        view->setRootIndex(selIndexes[0]);
}

void MainWindow::about()
{
   QMessageBox::about(this, tr("About PLaSK GUI"),
            tr("The <b>Dock Widgets</b> example demonstrates how to "
               "use Qt's dock widgets. You can enter your own text, "
               "click a customer to add a customer name and "
               "address, and click standard paragraphs to add them."));
}

void MainWindow::treeSelectionChanged(const QItemSelection & selected, const QItemSelection & deselected) {
    if (selected.indexes().empty()) {
        document.selectObject(0);  //deselect
    } else
        document.selectObject((GeometryTreeItem*) selected.indexes().first().internalPointer());
}

void MainWindow::treeRemoveSelected() {
    QModelIndexList s = treeView->selectionModel()->selectedRows();
    if (s.empty()) return;
    document.treeModel.removeRow(s[0].row(), s[0].parent());
}

void MainWindow::treeAddCartesian2d() {
    document.treeModel.appendGeometry(plask::make_shared<plask::Geometry2DCartesian>(plask::make_shared<plask::Extrusion>()));
}

void MainWindow::treeAddCartesian3d() {
        document.treeModel.appendGeometry(plask::make_shared<plask::Geometry3D>());
}

void MainWindow::treeAddCylindric() {
        document.treeModel.appendGeometry(plask::make_shared<plask::Geometry2DCylindrical>(plask::make_shared<plask::Revolution>()));
}

void MainWindow::treeAddBlock2D() {
    QModelIndexList s = treeView->selectionModel()->selectedRows();
    if (s.empty()) return;
    document.treeModel.insertRow(plask::make_shared<plask::Block<2> >(plask::vec(1.0, 1.0)), s[0]);
}

void MainWindow::createActions()
{
    newDocumentAct = new QAction(QIcon::fromTheme("document-new"), tr("&New"), this);
    newDocumentAct->setShortcuts(QKeySequence::New);
    newDocumentAct->setStatusTip(tr("Create a new document"));
    connect(newDocumentAct, &QAction::triggered, this, &MainWindow::newDocument);

    openAct = new QAction(QIcon::fromTheme("document-open"), tr("&Open..."), this);
    openAct->setShortcuts(QKeySequence::Open);
    openAct->setStatusTip(tr("Open experiment file"));
    connect(openAct, &QAction::triggered, this, &MainWindow::open);

    saveAct = new QAction(QIcon::fromTheme("document-save"), tr("&Save..."), this);
    saveAct->setShortcuts(QKeySequence::Save);
    saveAct->setStatusTip(tr("Save the current form letter"));
    connect(saveAct, &QAction::triggered, this, &MainWindow::save);

    printAct = new QAction(QIcon::fromTheme("document-print"), tr("&Print..."), this);
    printAct->setShortcuts(QKeySequence::Print);
    printAct->setStatusTip(tr("Print the current form letter"));
    connect(printAct, &QAction::triggered, this, &MainWindow::print);

    undoAct = new QAction(QIcon::fromTheme("edit-undo"), tr("&Undo"), this);
    undoAct->setShortcuts(QKeySequence::Undo);
    undoAct->setStatusTip(tr("Undo the last editing action"));
    connect(undoAct, &QAction::triggered, this, &MainWindow::undo);

    editSelectedAct = new QAction(QIcon::fromTheme("media-record"), tr("&Edit selected"), this);
    editSelectedAct->setStatusTip(tr("Edit selected object"));
    connect(editSelectedAct, &QAction::triggered, this, &MainWindow::editSelected);

    quitAct = new QAction(tr("&Quit"), this);
    quitAct->setShortcuts(QKeySequence::Quit);
    quitAct->setStatusTip(tr("Quit the application"));
    connect(quitAct, &QAction::triggered, this, &MainWindow::close);

    aboutAct = new QAction(tr("&About"), this);
    aboutAct->setStatusTip(tr("Show the application's About box"));
    connect(aboutAct, &QAction::triggered, this, &MainWindow::about);

    aboutQtAct = new QAction(tr("About &Qt"), this);
    aboutQtAct->setStatusTip(tr("Show the Qt library's About box"));
    connect(aboutQtAct, SIGNAL(triggered()), qApp, SLOT(aboutQt()));

    //tree actions:
    
    treeRemoveAct = new QAction(tr("&Remove"), this);
    treeRemoveAct->setToolTip(tr("Remove selected object from geometry tree"));
    connect(treeRemoveAct, &QAction::triggered, this, &MainWindow::treeRemoveSelected);
    
    treeAddCartesian2dAct = new QAction(tr("&New 2D cartesian geometry"), this);
    treeAddCartesian2dAct->setToolTip(tr("Create new, top-level, 2D cartesian geometry"));
    connect(treeAddCartesian2dAct, &QAction::triggered, this, &MainWindow::treeAddCartesian2d);
    
    treeAddCartesian3dAct = new QAction(tr("&New 3D cartesian geometry"), this);
    treeAddCartesian3dAct->setToolTip(tr("Create new, top-level, 3D cartesian geometry"));
    connect(treeAddCartesian3dAct, &QAction::triggered, this, &MainWindow::treeAddCartesian3d);

    treeAddCylindricAct = new QAction(tr("&New 2D cylindrical geometry"), this);
    treeAddCylindricAct->setToolTip(tr("Create new, top-level, 2D cylindric geometry"));
    connect(treeAddCylindricAct, &QAction::triggered, this, &MainWindow::treeAddCylindric);
    
    treeAddBlockAct = new QAction(tr("Add &Block2D"), this);
    connect(treeAddBlockAct, &QAction::triggered, this, &MainWindow::treeAddBlock2D);

    zoomInAct = new QAction(QIcon::fromTheme("zoom-in"), tr("Zoom &in"), this);
    zoomInAct->setToolTip(tr("Increase view zoom"));
    connect(zoomInAct, &QAction::triggered, view, &ObjectViewer::zoomIn);

    zoomOutAct = new QAction(QIcon::fromTheme("zoom-out"), tr("Zoom &out"), this);
    zoomOutAct->setToolTip(tr("Decrease view zoom"));
    connect(zoomOutAct, &QAction::triggered, view, &ObjectViewer::zoomOut);
}

void MainWindow::createMenus()
{
    fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(newDocumentAct);
    fileMenu->addAction(openAct);
    fileMenu->addAction(saveAct);
    fileMenu->addAction(printAct);
    fileMenu->addSeparator();
    fileMenu->addAction(quitAct);

    editMenu = menuBar()->addMenu(tr("&Edit"));
    editMenu->addAction(undoAct);
    editMenu->addAction(editSelectedAct);
    
    geometryMenu = menuBar()->addMenu(tr("&Geometry"));
    geometryMenu->addAction(treeAddCartesian2dAct);
    geometryMenu->addAction(treeAddCylindricAct);
    geometryMenu->addAction(treeAddCartesian3dAct);
    geometryMenu->addSeparator();
    geometryMenu->addAction(treeRemoveAct);
    

    viewMenu = menuBar()->addMenu(tr("&View"));

    menuBar()->addSeparator();

    helpMenu = menuBar()->addMenu(tr("&Help"));
    helpMenu->addAction(aboutAct);
    helpMenu->addAction(aboutQtAct);
}

void MainWindow::createToolBars()
{
    fileToolBar = addToolBar(tr("File"));
    fileToolBar->addAction(newDocumentAct);
    fileToolBar->addAction(openAct);
    fileToolBar->addAction(saveAct);
    fileToolBar->addAction(printAct);

    editToolBar = addToolBar(tr("Edit"));
    editToolBar->addAction(undoAct);
    editToolBar->addAction(editSelectedAct);

    viewToolBar = addToolBar(tr("View"));
    viewToolBar->addAction(zoomInAct);
    viewToolBar->addAction(zoomOutAct);
}

void MainWindow::createStatusBar()
{
    statusBar()->showMessage(tr("Ready"));
}

void MainWindow::createDockWindows() {
    QDockWidget *dock = new QDockWidget(tr("Geometry objects tree"), this);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea); 
    treeView = new QTreeView(dock);
    treeView->setAlternatingRowColors(true);    //2 colors for even/odd
    treeView->setContextMenuPolicy(Qt::ActionsContextMenu);
    treeView->addAction(treeRemoveAct);
    treeView->addAction(treeAddBlockAct);
    treeView->setModel(&document.treeModel);
    //treeView->setDragEnabled(true);
    //treeView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeView->setDragEnabled(true);
    treeView->setAcceptDrops(true);
    treeView->setDropIndicatorShown(true);
    treeView->setDragDropMode(QAbstractItemView::DragDrop);
    //treeView->selectionModel()->
    QObject::connect(treeView->selectionModel(), SIGNAL(selectionChanged(const QItemSelection &, const QItemSelection &)),
                     this, SLOT(treeSelectionChanged(const QItemSelection &, const QItemSelection &)));
    dock->setWidget(treeView);
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    viewMenu->addAction(dock->toggleViewAction());

    dock = new QDockWidget(tr("Properties"), this);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
   // propertyTree = new QtTreePropertyBrowser(dock);
    /*QWidget* group = new QWidget(this);
    QFormLayout* l = new QFormLayout(group);
    group->setLayout(l);
    l->addRow("ExampleLabel1", new QSpinBox);
    l->addRow("ExampleLabel2", new QSpinBox);
    dock->setWidget(group);*/
    dock->setWidget(propertyTree);
    addDockWidget(Qt::LeftDockWidgetArea, dock);
    viewMenu->addAction(dock->toggleViewAction());

    dock = new QDockWidget(tr("New geometry objects"), this);
    dock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    creatorsList = new QListView(dock);
    creatorsList->setModel(&creators);
    creatorsList->setDragDropMode(QAbstractItemView::DragOnly);
    dock->setWidget(creatorsList);
    addDockWidget(Qt::RightDockWidgetArea, dock);
    viewMenu->addAction(dock->toggleViewAction());
}

