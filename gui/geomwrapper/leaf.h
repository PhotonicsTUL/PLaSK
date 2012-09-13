#ifndef GUI_GEOMETRY_WRAPPER_LEAF_H
#define GUI_GEOMETRY_WRAPPER_LEAF_H

/** @file
 * This file includes implementation of geometry objects model extensions for leafs. Do not include it directly (see register.h).
 */

#include "element.h"

#include <plask/geometry/leaf.h>

template <int dim>
struct BlockWrapper: public ObjectWrapperFor< plask::Block<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

};

#endif // LEAF_H
