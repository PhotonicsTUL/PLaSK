#ifndef PLASK_GUI_DOCUMENT_H
#define PLASK_GUI_DOCUMENT_H

#include <plask/manager.h>
#include <QString>
#include <QUndoStack>

#include "tree.h"

#include "utils/propbrowser.h"

/**
 * Represent document with experiment description.
 * Includes geometry.
 */
class Document
{

public:

    GeometryTreeModel treeModel;

    BrowserWithManagers propertiesBrowser;

    QUndoStack undoStack;

    void open(const QString& fileName);

    Document(QtAbstractPropertyBrowser& browser);

    void selectElement(GeometryTreeItem* treeItem) {
        propertiesBrowser.clear();
        if (treeItem == 0) return;
        treeItem->fillPropertyBrowser(propertiesBrowser);
    }
    
    void clear();
};

#endif // PLASK_GUI_DOCUMENT_H
