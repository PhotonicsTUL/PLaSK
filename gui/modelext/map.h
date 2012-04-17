#ifndef PLASK_GUI_MODEL_EXT_MAP_H
#define PLASK_GUI_MODEL_EXT_MAP_H

/** @file
 * This file includes interface to plask geometry elements model extensions connected with Qt.
 *
 * Typically you should call ext() function with reference to your geometry element object as argument.
 * This function return ElementExtension which provide methods which operates on your element, knowing its type.
 */

#include <QPainter>
#include <QGraphicsItem>
#include "../utils/propbrowser.h"

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
class QRectF;
QT_END_NAMESPACE

#include <plask/geometry/element.h>

/**
 * Base class for objects which cast geometry element to conrate type and calls its methods.
 */
struct ElementExtensionImplBase {

    virtual ~ElementExtensionImplBase();

    /**
     * Draw geometry element using given Qt @p painter.
     * @param toDraw element to draw
     * @param painter where draw element
     */
    virtual void draw(const plask::GeometryElement& toDraw, QPainter& painter) const;

    /**
     * Draw miniature with size close to given.
     * @param toDraw element for which miniature should be drawn
     * @param painter where miniature shpuld be drawn
     * @param w, h requested miniature size
     */
    virtual void drawMiniature(const plask::GeometryElement& toDraw, QPainter& painter, qreal w, qreal h) const;

    /**
     * Get miniature image with size close to given.
     * @param toDraw element for which miniature should be drawn
     * @param w, h requested miniature size
     * @return miniature
     */
    QPixmap getMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h) const;

    /**
     * Get string representation of given element.
     * @param el geometry element
     * @return string representation of @p el, can have multiple lines of text
     */
    virtual QString toStr(const plask::GeometryElement& el) const;

    /**
     * Fill property browser with properties of @p el.
     */
    virtual void setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

    void setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers) const {
        setupPropertiesBrowser(el, managers, managers.browser);
    }

    /**
     * Fill property browser with properties of @p container child.
     *
     * This is called only for containers and default implementation call setupPropertiesBrowser for pointed child.
     * Typically, you can call ElementExtensionImplBase::setupPropertiesBrowserForChild in subclasses.
     * @param container geometry element
     * @param index real child index
     */
    virtual void setupPropertiesBrowserForChild(plask::GeometryElement& container, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

    void setupPropertiesBrowserForChild(plask::GeometryElement& container, std::size_t index, BrowserWithManagers& managers) const {
        setupPropertiesBrowserForChild(container, index, managers, managers.browser);
    }

};

/**
 * Wrap geometry element instance and allow to call its extra methods.
 *
 * Typically you should call ext(const plask::GeometryElement&) function to get instance of this class.
 *
 * It is safe and fast to pass ElementExtension instances by value.
 */
struct ElementExtension {

    const ElementExtensionImplBase& impl;

    plask::GeometryElement& element;

    ElementExtension(const ElementExtensionImplBase& impl, plask::GeometryElement& element)
        : impl(impl), element(element) {}

    //delegators:

    void draw(QPainter& painter) const { impl.draw(element, painter); }

    QString toStr() const {  return impl.toStr(element); }

    QPixmap getMiniature(qreal w, qreal h) const { return impl.getMiniature(element, w, h); }

    void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const { impl.setupPropertiesBrowser(element, managers, dst); }

    void setupPropertiesBrowser(BrowserWithManagers& managers) const { impl.setupPropertiesBrowser(element, managers); }

    void setupPropertiesBrowserForChild(std::size_t childIndex, BrowserWithManagers &managers, QtAbstractPropertyBrowser& dst) const { impl.setupPropertiesBrowserForChild(element, childIndex, managers, dst); }

    void setupPropertiesBrowserForChild(std::size_t childIndex, BrowserWithManagers &managers) const { impl.setupPropertiesBrowserForChild(element, childIndex, managers); }

};

/// Initialize model extensions mechanism. You should call this once before calling ext(const plask::GeometryElement&).
void initModelExtensions();

/**
 * Get extension for geometry element @p el.
 * @param el geometry element
 * @return extension for real type of @p el
 */
ElementExtension ext(plask::GeometryElement& el);

/**
 * Get extension for geometry element @p el (const version).
 * @param el (const) geometry element
 * @return extension for real type of @p el
 */
inline ElementExtension ext(const plask::GeometryElement& el) {
    return ext(const_cast<plask::GeometryElement&>(el));
}

#endif // PLASK_GUI_MODEL_EXT_MAP_H
