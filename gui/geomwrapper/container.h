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

};

template <int dim>
struct MultiStackWrapper: public ElementWrapperFor< plask::MultiStackContainer<dim>, StackWrapper<dim> > {

    virtual QString toStr() const;

    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst);

    //virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

};

#endif // CONTAINER_H
