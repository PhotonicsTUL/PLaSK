#ifndef GUI_GEOMETRY_WRAPPER_TRANSFORM_H
#define GUI_GEOMETRY_WRAPPER_TRANSFORM_H

/** @file
 * This file includes implementation of geometry objects model extensions for transforms. Do not include it directly (see register.h).
 */

#include "object.h"

#include <plask/geometry/transform.h>
#include <plask/geometry/transform_space_cartesian.h>

template <int dim>
struct TranslationWrapper: public ObjectWrapperFor< plask::Translation<dim> > {

    /**
     * Names of path hints, used only if this translation is inside container.
     */
    std::vector<std::string> pathHintsNames;
    
    virtual QString toStr() const;

    virtual void draw(QPainter& painter, bool paintBorders) const;

    virtual bool canInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) const;

    virtual bool tryInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index);

};

struct ExtrusionWrapper: public ObjectWrapperFor< plask::Extrusion > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

};

#endif //GUI_GEOMETRY_WRAPPER_TRANSFORM_H
