#ifndef PLASK_GUI_DOCUMENT_H
#define PLASK_GUI_DOCUMENT_H

#include <plask/geometry/manager.h>

/**
 * Represent document with experiment description.
 * Includes geometry.
 */
class Document
{

    plask::GeometryManager manager;

public:
    Document();
};

#endif // PLASK_GUI_DOCUMENT_H
