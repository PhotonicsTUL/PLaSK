#ifndef GUI_GEOMETRY_WRAPPER_GEOMETRY_H
#define GUI_GEOMETRY_WRAPPER_GEOMETRY_H

/** @file
 * This file includes implementation of geometry objects model extensions for geometries. Do not include it directly (see register.h).
 */

#include "object.h"

#include <plask/geometry/space.h>

template <int dim>
struct GeometryWrapper: public ObjectWrapperFor< plask::GeometryD<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    // delegate to child:

    virtual void draw(QPainter& painter) const;

    virtual void drawMiniature(QPainter& painter, qreal w, qreal h, bool saveProp) const;

    virtual void drawReal(QPainter& painter) const;

    //TODO for insertion impl. see translation
};

struct Geometry2DCartesianWrapper: public GeometryWrapper<2> {

    typedef plask::Geometry2DCartesian WrappedType;

    virtual QString toStr() const;
    
    plask::shared_ptr<plask::Extrusion> getExtrusion() const;
    
    plask::Geometry2DCartesian& getCartesian2D() const;
    
    virtual bool canInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) const;
    
    virtual bool canInsert(const GeometryObjectCreator& to_insert, std::size_t index) const;
    
    virtual bool tryInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index);
    
    virtual bool tryInsert(const GeometryObjectCreator& to_insert, std::size_t index);

};

struct Geometry2DCylindricalWrapper: public GeometryWrapper<2> {

    typedef plask::Geometry2DCylindrical WrappedType;

    virtual QString toStr() const;
    
    plask::shared_ptr<plask::Revolution> getRevolution() const;
    
    plask::Geometry2DCylindrical& getCylindrical2D() const;
    
    virtual bool canInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) const;
    
    virtual bool canInsert(const GeometryObjectCreator& to_insert, std::size_t index) const;
    
    virtual bool tryInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index);
    
    virtual bool tryInsert(const GeometryObjectCreator& to_insert, std::size_t index);

};

#endif // GEOMETRY_H
