"""
/****************************************************************************
**
** Copyright (C) 2012 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** Copyright (C) 2012 Jerome Souquieres
** Adapted to Python from ModelTest.cpp from the test suite of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** GNU Lesser General Public License Usage
** This file may be used under the terms of the GNU Lesser General Public
** License version 2.1 as published by the Free Software Foundation and
** appearing in the file LICENSE.LGPL included in the packaging of this
** file. Please review the following information to ensure the GNU Lesser
** General Public License version 2.1 requirements will be met:
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights. These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU General
** Public License version 3.0 as published by the Free Software Foundation
** and appearing in the file LICENSE.GPL included in the packaging of this
** file. Please review the following information to ensure the GNU General
** Public License version 3.0 requirements will be met:
** http://www.gnu.org/copyleft/gpl.html.
**
** Other Usage
** Alternatively, this file may be used in accordance with the terms and
** conditions contained in a signed written agreement between you and Nokia.
**
**
**
**
**
** $QT_END_LICENSE$
**
****************************************************************************/

Based on http://qt.gitorious.org/qt/qt/blobs/4.8/tests/auto/modeltest

Repository:qt
Project:Qt
Owner:The Qt Project
Branch: 4.8
HEAD:1af0087
HEAD tree:bddde76

"""

from ..qt.QtCore import *


def QCOMPARE(v1, v2):
    """@todo: quick&dirty for now"""
    if not v1 == v2:
        raise Exception("QCOMPARE")


def QVERIFY(v):
    """@todo: quick&dirty for now"""
    if not v:
        raise Exception("QVERIFY")


class Changing(object):
    def __init__(self):
        self.parent = None
        self.oldSize = 0
        self.last = None
        self.next = None


class ModelTest(QObject):

    def __init__(self, model, parent=None):
        """
        Connect to all of the models signals.  Whenever anything happens recheck everything.
        """
        QObject.__init__(self, parent)
        self._model = model
        self._fetchingMore = False
        self._insert = []
        self._remove = []
        self._changing = []

        if not self._model:
            raise Exception("model must not be null")

        self._model.columnsAboutToBeInserted.connect(self.runAllTests)
        self._model.columnsAboutToBeRemoved.connect(self.runAllTests)
        self._model.columnsInserted.connect(self.runAllTests)
        self._model.columnsRemoved.connect(self.runAllTests)
        self._model.dataChanged.connect(self.runAllTests)
        self._model.headerDataChanged.connect(self.runAllTests)
        self._model.layoutAboutToBeChanged.connect(self.runAllTests)
        self._model.layoutChanged.connect(self.runAllTests)
        self._model.modelReset.connect(self.runAllTests)
        self._model.rowsAboutToBeInserted.connect(self.runAllTests)
        self._model.rowsAboutToBeRemoved.connect(self.runAllTests)
        self._model.rowsInserted.connect(self.runAllTests)
        self._model.rowsRemoved.connect(self.runAllTests)

        # Special checks for inserting/removing
        self._model.layoutAboutToBeChanged.connect(self.layoutAboutToBeChanged)
        self._model.layoutChanged.connect(self.layoutChanged)
        self._model.layoutChanged.connect(self.layoutChanged)
        self._model.rowsAboutToBeInserted.connect(self.rowsAboutToBeInserted)
        self._model.rowsAboutToBeRemoved.connect(self.rowsAboutToBeRemoved)
        self._model.rowsInserted.connect(self.rowsInserted)
        self._model.rowsRemoved.connect(self.rowsRemoved)

        self.runAllTests()

    def runAllTests(self):
        if self._fetchingMore:
            return
        self.nonDestructiveBasicTest()
        self.rowCount()
        self.columnCount()
        self.hasIndex()
        self.index()
        self.parent()
        self.data()

    def nonDestructiveBasicTest(self):
        """
        nonDestructiveBasicTest tries to call a number of the basic functions (not all)
        to make sure the model doesn't outright segfault, testing the functions that makes sense.
        """
        QVERIFY( self._model.buddy(QModelIndex()) == QModelIndex() )
        self._model.canFetchMore ( QModelIndex() )
        QVERIFY( self._model.columnCount ( QModelIndex() ) >= 0 )
        QVERIFY( self._model.data( QModelIndex(), Qt.DisplayRole ) == None )
        self._fetchingMore = True
        self._model.fetchMore ( QModelIndex() )
        self._fetchingMore = False
        flags = self._model.flags ( QModelIndex() )
        QVERIFY( flags == Qt.ItemIsDropEnabled or flags == 0 )
        self._model.hasChildren ( QModelIndex() )
        self._model.hasIndex ( 0, 0 )
        self._model.headerData ( 0, Qt.Horizontal, Qt.DisplayRole )
        self._model.index ( 0, 0, QModelIndex() )
        self._model.itemData ( QModelIndex() )
        cache = None
        self._model.match ( QModelIndex(), -1, cache )
        self._model.mimeTypes()
        self._model.parent ( QModelIndex() )
        QVERIFY( self._model.parent ( QModelIndex() ) == QModelIndex() )
        QVERIFY( self._model.rowCount(QModelIndex()) >= 0 )
        variant = None
        self._model.setData ( QModelIndex(), variant, -1 )
        self._model.setHeaderData ( -1, Qt.Horizontal, variant )
        self._model.setHeaderData ( 999999, Qt.Horizontal, variant )
        self._model.sibling ( 0, 0, QModelIndex() )
        self._model.span ( QModelIndex() )
        self._model.supportedDropActions()

    def rowCount(self):
        """
        Tests model's implementation of QAbstractItemModel::rowCount() and hasChildren()

        Models that are dynamically populated are not as fully tested here.
        """
        #     qDebug() << "rc"
        # check top row
        topIndex = self._model.index ( 0, 0, QModelIndex() )
        rows = self._model.rowCount ( topIndex )
        QVERIFY( rows >= 0 )
        if rows > 0:
            QVERIFY( self._model.hasChildren ( topIndex ) )

        secondLevelIndex = self._model.index ( 0, 0, topIndex )
        if secondLevelIndex.isValid():  #  not the top level
            # check a row count where parent is valid
            rows = self._model.rowCount ( secondLevelIndex )
            QVERIFY( rows >= 0 )
            if rows > 0:
                QVERIFY( self._model.hasChildren ( secondLevelIndex ) )

        # The models rowCount() is tested more extensively in checkChildren(),
        # but this catches the big mistakes


    def columnCount(self):
        """
        Tests model's implementation of QAbstractItemModel::columnCount() and hasChildren()
        """
        # check top row
        topIndex = self._model.index ( 0, 0, QModelIndex() )
        QVERIFY( self._model.columnCount ( topIndex ) >= 0 )

        # check a column count where parent is valid
        childIndex = self._model.index ( 0, 0, topIndex )
        if childIndex.isValid():
            QVERIFY( self._model.columnCount ( childIndex ) >= 0 )

        # columnCount() is tested more extensively in checkChildren(),
        # but this catches the big mistakes

    def hasIndex(self):
        """
        Tests model's implementation of QAbstractItemModel::hasIndex()
        """
        #     qDebug() << "hi"
        # Make sure that invalid values returns an invalid index
        QVERIFY( not self._model.hasIndex ( -2, -2 ) )
        QVERIFY( not self._model.hasIndex ( -2, 0 ) )
        QVERIFY( not self._model.hasIndex ( 0, -2 ) )

        rows = self._model.rowCount(QModelIndex())
        columns = self._model.columnCount(QModelIndex())

        # check out of bounds
        QVERIFY( not self._model.hasIndex ( rows, columns ) )
        QVERIFY( not self._model.hasIndex ( rows + 1, columns + 1 ) )

        if rows > 0:
            QVERIFY( self._model.hasIndex ( 0, 0 ) )

        # hasIndex() is tested more extensively in checkChildren(),
        # but this catches the big mistakes


    def index(self):
        """
        Tests model's implementation of QAbstractItemModel::index()
        """
        #     qDebug() << "i"
        # Make sure that invalid values returns an invalid index
        QVERIFY( self._model.index ( -2, -2, QModelIndex() ) == QModelIndex() )
        QVERIFY( self._model.index ( -2, 0, QModelIndex() ) == QModelIndex() )
        QVERIFY( self._model.index ( 0, -2, QModelIndex() ) == QModelIndex() )

        rows = self._model.rowCount(QModelIndex())
        columns = self._model.columnCount(QModelIndex())

        if rows == 0:
            return

        # Catch off by one errors
        QVERIFY( self._model.index ( rows, columns, QModelIndex() ) == QModelIndex() )
        QVERIFY( self._model.index ( 0, 0, QModelIndex() ).isValid() )

        # Make sure that the same index is *always* returned
        a = self._model.index ( 0, 0, QModelIndex() )
        b = self._model.index ( 0, 0, QModelIndex() )
        QVERIFY( a == b )

        # index() is tested more extensively in checkChildren(),
        # but this catches the big mistakes


    def parent(self):
        """
        Tests model's implementation of QAbstractItemModel::parent()
        """
        #     qDebug() << "p"
        # Make sure the model wont crash and will return an invalid QModelIndex
        # when asked for the parent of an invalid index.
        QVERIFY( self._model.parent ( QModelIndex() ) == QModelIndex() )

        if self._model.rowCount(QModelIndex()) == 0:
            return

        # Column 0                | Column 1    |
        # QModelIndex()           |             |
        #    \- topIndex          | topIndex1   |
        #         \- childIndex   | childIndex1 |

        # Common error test #1, make sure that a top level index has a parent
        # that is a invalid QModelIndex.
        topIndex = self._model.index ( 0, 0, QModelIndex() )
        QVERIFY( self._model.parent ( topIndex ) == QModelIndex() )

        # Common error test #2, make sure that a second level index has a parent
        # that is the first level index.
        if self._model.rowCount ( topIndex ) > 0:
            childIndex = self._model.index ( 0, 0, topIndex )
            QVERIFY( self._model.parent ( childIndex ) == topIndex )


        # Common error test #3, the second column should NOT have the same children
        # as the first column in a row.
        # Usually the second column shouldn't have children.
        topIndex1 = self._model.index ( 0, 1, QModelIndex() )
        if self._model.rowCount ( topIndex1 ) > 0:
            childIndex = self._model.index ( 0, 0, topIndex )
            childIndex1 = self._model.index ( 0, 0, topIndex1 )
            QVERIFY( childIndex != childIndex1 )


        # Full test, walk n levels deep through the model making sure that all
        # parent's children correctly specify their parent.
        self.checkChildren ( QModelIndex() )


    def checkChildren(self, parent, currentDepth=0):
        """
        Called from the parent() test.

        A model that returns an index of parent X should also return X when asking
        for the parent of the index.

        This recursive function does pretty extensive testing on the whole model in an
        effort to catch edge cases.

        This function assumes that rowCount(), columnCount() and index() already work.
        If they have a bug it will point it out, but the above tests should have already
        found the basic bugs because it is easier to figure out the problem in
        those tests then this one.
        """
        # First just try walking back up the tree.
        p = parent
        while p.isValid():
            p = p.parent()

        # For models that are dynamically populated
        if self._model.canFetchMore ( parent ):
            self._fetchingMore = True
            self._model.fetchMore ( parent )
            self._fetchingMore = False

        rows = self._model.rowCount ( parent )
        columns = self._model.columnCount ( parent )

        if rows > 0:
            QVERIFY( self._model.hasChildren ( parent ) )

        # Some further testing against rows(), columns(), and hasChildren()
        QVERIFY( rows >= 0 )
        QVERIFY( columns >= 0 )
        if rows > 0:
            QVERIFY( self._model.hasChildren ( parent ) )

        #qDebug() << "parent:" << self._model.data(parent).toString() << "rows:" << rows
        #         << "columns:" << columns << "parent column:" << parent.column()

        QVERIFY( not self._model.hasIndex ( rows + 1, 0, parent ) )
        for r in range(rows):
            if self._model.canFetchMore ( parent ):
                self._fetchingMore = True
                self._model.fetchMore ( parent )
                self._fetchingMore = False

            QVERIFY( not self._model.hasIndex ( r, columns + 1, parent ) )
            for c in range(columns) :
                QVERIFY( self._model.hasIndex ( r, c, parent ) )
                index = self._model.index ( r, c, parent )
                # rowCount() and columnCount() said that it existed...
                QVERIFY( index.isValid() )

                # index() should always return the same index when called twice in a row
                modifiedIndex = self._model.index ( r, c, parent )
                QVERIFY( index == modifiedIndex )

                # Make sure we get the same index if we request it twice in a row
                a = self._model.index ( r, c, parent )
                b = self._model.index ( r, c, parent )
                QVERIFY( a == b )

                # Some basic checking on the index that is returned
                QVERIFY( index.model() == self._model )
                QCOMPARE( index.row(), r )
                QCOMPARE( index.column(), c )
                # While you can technically return a QVariant usually this is a sign
                # of a bug in data().  Disable if this really is ok in your model.
                # QVERIFY( self._model.data ( index, Qt.DisplayRole ).isValid() )

                # If the next test fails here is some somewhat useful debug you play with.

                if self._model.parent(index) != parent:
                    #qDebug() << r << c << currentDepth << self._model.data(index).toString()
                    #         << self._model.data(parent).toString()
                    #qDebug() << index << parent << self._model.parent(index)
                    #   And a view that you can even use to show the model.
                    #   QTreeView view
                    #   view.setModel(model)
                    #   view.show()
                    pass

                # Check that we can get back our real parent.
                QCOMPARE( self._model.parent ( index ), parent )

                # recursively go down the children
                if self._model.hasChildren ( index ) and currentDepth < 10:
                    #qDebug() << r << c << "has children" << self._model.rowCount(index)
                    self.checkChildren ( index, ++currentDepth )
                # /* else { if (currentDepth >= 10) qDebug() << "checked 10 deep"; };*/

                # make sure that after testing the children that the index doesn't change.
                newerIndex = self._model.index ( r, c, parent )
                QVERIFY( index == newerIndex )

    def data(self):
        """
        Tests model's implementation of QAbstractItemModel::data()
        """
        # Invalid index should return an invalid qvariant
        QVERIFY( not self._model.data( QModelIndex(), Qt.DisplayRole ) )

        if self._model.rowCount(QModelIndex()) == 0:
            return

        # A valid index should have a valid QVariant data
        QVERIFY( self._model.index ( 0, 0, QModelIndex() ).isValid() )

        # shouldn't be able to set data on an invalid index
        QVERIFY( not self._model.setData ( QModelIndex(), "foo", Qt.DisplayRole ) )

        # General Purpose roles that should return a QString
        variant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.ToolTipRole )
        if variant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QString> ( variant ) )

        variant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.StatusTipRole )
        if variant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QString> ( variant ) )

        variant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.WhatsThisRole )
        if variant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QString> ( variant ) )


        # General Purpose roles that should return a QSize
        variant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.SizeHintRole )
        if variant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QSize> ( variant ) )

        # General Purpose roles that should return a QFont
        fontVariant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.FontRole )
        if fontVariant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QFont> ( fontVariant ) )

        # Check that the alignment is one we know about
        textAlignmentVariant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.TextAlignmentRole )
        if textAlignmentVariant:
            pass # TODO
            # alignment = textAlignmentVariant.toInt()
            # QCOMPARE( alignment, ( alignment & ( Qt.AlignHorizontal_Mask | Qt.AlignVertical_Mask ) ) )

        # General Purpose roles that should return a QColor
        colorVariant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.BackgroundColorRole )
        if colorVariant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QColor> ( colorVariant ) )

        colorVariant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.TextColorRole )
        if colorVariant:
            pass # TODO
            # QVERIFY( qVariantCanConvert<QColor> ( colorVariant ) )

        # Check that the "check state" is one we know about.
        checkStateVariant = self._model.data( self._model.index ( 0, 0, QModelIndex() ), Qt.CheckStateRole )
        if checkStateVariant:
            pass # TODO
            # state = checkStateVariant.toInt()
            # QVERIFY( state == Qt.Unchecked or
            #          state == Qt.PartiallyChecked or
            #          state == Qt.Checked )

    def rowsAboutToBeInserted (self, parent, start, end):
        """
        Store what is about to be inserted to make sure it actually happens

        \sa rowsInserted()
        """

        #     Q_UNUSED(end)
        #    qDebug() << "rowsAboutToBeInserted" << "start=" << start << "end=" << end << "parent=" << self._model.data ( parent ).toString()
        #    << "current count of parent=" << self._model.rowCount ( parent ); # << "display of last=" << self._model.data( self._model.index(start-1, 0, parent) )
        #     qDebug() << self._model.index(start-1, 0, parent) << self._model.data( self._model.index(start-1, 0, parent) )
        c = Changing()
        c.parent = parent
        c.oldSize = self._model.rowCount ( parent )
        c.last = self._model.data ( self._model.index ( start - 1, 0, parent ), Qt.DisplayRole )
        c.next = self._model.data ( self._model.index ( start, 0, parent ), Qt.DisplayRole )
        self._insert.append(c)


    def rowsInserted(self, parent, start, end):
        """
        Confirm that what was said was going to happen actually did

        \sa rowsAboutToBeInserted()
        """
        c = self._insert.pop()
        QVERIFY( c.parent == parent )
        #    qDebug() << "rowsInserted"  << "start=" << start << "end=" << end << "oldsize=" << c.oldSize
        #    << "parent=" << self._model.data ( parent ).toString() << "current rowcount of parent=" << self._model.rowCount ( parent )

        #    for (int ii=start; ii <= end; ii++)
        #    {
        #      qDebug() << "itemWasInserted:" << ii << self._model.data ( self._model.index ( ii, 0, parent ))
        #    }
        #    qDebug()

        QVERIFY( c.oldSize + ( end - start + 1 ) == self._model.rowCount ( parent ) )
        QVERIFY( c.last == self._model.data( self._model.index ( start - 1, 0, c.parent ), Qt.DisplayRole ) )

        if c.next != self._model.data(self._model.index(end + 1, 0, c.parent), Qt.DisplayRole):
            #qDebug() << start << end
            #for (int i=0; i < self._model.rowCount(); ++i)
            #    qDebug() << self._model.index(i, 0).data().toString()
            #qDebug() << c.next << self._model.data(self._model.index(end + 1, 0, c.parent))
            pass


        QVERIFY( c.next == self._model.data( self._model.index ( end + 1, 0, c.parent ), Qt.DisplayRole ) )

    def layoutAboutToBeChanged(self):
        for i in range(min(max(0, self._model.rowCount(QModelIndex())), 100)):
            self._changing.append ( QPersistentModelIndex ( self._model.index ( i, 0, QModelIndex() ) ) )

    def layoutChanged(self):
        for p in self._changing:
            QVERIFY( p == self._model.index ( p.row(), p.column(), p.parent() ) )
        self._changing[:] = []



    def rowsAboutToBeRemoved(self, parent, start, end):
        """
        Store what is about to be inserted to make sure it actually happens

        \sa rowsRemoved()
        """
        # qDebug() << "ratbr" << parent << start << end
        c = Changing()
        c.parent = parent
        c.oldSize = self._model.rowCount ( parent )
        c.last = self._model.data(self._model.index ( start - 1, 0, parent ), Qt.DisplayRole)
        c.next = self._model.data(self._model.index ( end + 1, 0, parent ), Qt.DisplayRole)
        self._remove.append( c )

    def rowsRemoved(self, parent, start, end):
        """
        Confirm that what was said was going to happen actually did

        \sa rowsAboutToBeRemoved()
        """
        #qDebug() << "rr" << parent << start << end
        c = self._remove.pop()
        QVERIFY(c.parent == parent )
        QVERIFY(c.oldSize - ( end - start + 1 ) == self._model.rowCount( parent ) )
        QVERIFY(c.last == self._model.data(self._model.index ( start - 1, 0, c.parent ), Qt.DisplayRole))
        QVERIFY(c.next == self._model.data(self._model.index ( start, 0, c.parent ), Qt.DisplayRole))