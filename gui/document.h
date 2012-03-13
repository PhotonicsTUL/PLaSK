#ifndef PLASK_GUI_DOCUMENT_H
#define PLASK_GUI_DOCUMENT_H

#include <plask/geometry/manager.h>
#include <QString>

/**
 * Represent document with experiment description.
 * Includes geometry.
 */
class Document
{

public:

    plask::GeometryManager manager;

    void open(const QString& fileName);

    Document();
};

#endif // PLASK_GUI_DOCUMENT_H
