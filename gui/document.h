#ifndef PLASK_GUI_DOCUMENT_H
#define PLASK_GUI_DOCUMENT_H

#include <plask/geometry/manager.h>
#include <QString>
#include "tree.h"

/**
 * Represent document with experiment description.
 * Includes geometry.
 */
class Document
{

public:

    plask::GeometryManager manager;

    GeometryTreeModel treeModel;

    void open(const QString& fileName);

    Document();
};

#endif // PLASK_GUI_DOCUMENT_H
