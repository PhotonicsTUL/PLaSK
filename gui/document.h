#ifndef PLASK_GUI_DOCUMENT_H
#define PLASK_GUI_DOCUMENT_H

#include <plask/geometry/manager.h>
#include <QString>
#include "tree.h"

#include "utils/propbrowser.h"

/**
 * Represent document with experiment description.
 * Includes geometry.
 */
class Document
{

public:

    plask::GeometryManager manager;

    GeometryTreeModel treeModel;

    BrowserWithManagers propertiesBrowser;

    void open(const QString& fileName);

    Document(QtAbstractPropertyBrowser& browser);

    void selectElement(GeometryTreeItem* treeItem);
};

#endif // PLASK_GUI_DOCUMENT_H
