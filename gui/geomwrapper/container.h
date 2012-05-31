#ifndef GUI_GEOMETRY_WRAPPER_CONTAINER_H
#define GUI_GEOMETRY_WRAPPER_CONTAINER_H

/** @file
 * This file includes implementation of geometry elements model extensions for containers. Do not include it directly (see register.h).
 */

#include "element.h"

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>

template <int dim>
struct StackWrapper: public ElementWrapperFor< plask::StackContainer<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    //TODO can be move to generic container wrapper
    virtual bool canInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) const {
        return index <= this->c().getRealChildrenCount() && to_insert->getDimensionsCount() == dim && this->c().canHasAsChild(*to_insert);
    }

};

template <int dim>
struct MultiStackWrapper: public ElementWrapperFor< plask::MultiStackContainer<dim>, StackWrapper<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    //virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

};

#endif // CONTAINER_H
