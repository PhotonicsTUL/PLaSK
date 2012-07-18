#ifndef GUI_GEOMETRY_WRAPPER_GEOMETRY_H
#define GUI_GEOMETRY_WRAPPER_GEOMETRY_H

/** @file
 * This file includes implementation of geometry elements model extensions for geometries. Do not include it directly (see register.h).
 */

#include "element.h"

#include <plask/geometry/space.h>

template <int dim>
struct GeometryWrapper: public ElementWrapperFor< plask::GeometryD<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);


    // delegate to child:

    virtual void draw(QPainter& painter) const;

    virtual void drawMiniature(QPainter& painter, qreal w, qreal h) const;

    virtual void drawReal(QPainter& painter) const;
};

struct Geometry2DCartesianWrapper: public GeometryWrapper<2> {

    virtual QString toStr() const;

    typedef plask::Geometry2DCartesian WrappedType;

};

struct Geometry2DCylindricalWrapper: public GeometryWrapper<2> {

    virtual QString toStr() const;

    typedef plask::Geometry2DCylindrical WrappedType;

};

#endif // GEOMETRY_H
