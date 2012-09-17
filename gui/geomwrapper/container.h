#ifndef GUI_GEOMETRY_WRAPPER_CONTAINER_H
#define GUI_GEOMETRY_WRAPPER_CONTAINER_H

/** @file
 * This file includes implementation of geometry objects model extensions for containers. Do not include it directly (see register.h).
 */

#include "object.h"

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>

template <int dim>
struct StackWrapper: public ObjectWrapperFor< plask::StackContainer<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    virtual bool tryInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) {
        if (!this->canInsert(to_insert, index)) return false;
        this->c().insertUnsafe(plask::static_pointer_cast< plask::GeometryObjectD<dim> >(to_insert), index);
        return true;
    }

    virtual int getInsertionIndexForPoint(const plask::Vec<2, double>& point);

    plask::Box2D getInsertPlace2D(const GeometryObjectCreator &to_insert, const plask::Vec<2, double> &point);

};

template <int dim>
struct MultiStackWrapper: public ObjectWrapperFor< plask::MultiStackContainer<dim>, StackWrapper<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    //virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

};

struct ShelfWrapper: public ObjectWrapperFor< plask::ShelfContainer2D > {

    virtual QString toStr() const;

    virtual bool canInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) const {
        return index <= this->c().getRealChildrenCount() && to_insert->getDimensionsCount() == 2 && this->c().canHasAsChild(*to_insert);
    }

    virtual bool tryInsert(plask::shared_ptr<plask::GeometryObject> to_insert, std::size_t index) {
        if (!canInsert(to_insert, index)) return false;
        this->c().insertUnsafe(plask::static_pointer_cast< plask::GeometryObjectD<2> >(to_insert), index);
        return true;
    }

    virtual int getInsertionIndexForPoint(const plask::Vec<2, double>& point);

    plask::Box2D getInsertPlace2D(const GeometryObjectCreator &to_insert, const plask::Vec<2, double> &point);

};

#endif // CONTAINER_H
