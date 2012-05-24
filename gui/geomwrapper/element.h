#ifndef GUI_GEOMETRY_WRAPPER_ELEMENT_H
#define GUI_GEOMETRY_WRAPPER_ELEMENT_H

/** @file
 * This file includes interface to plask geometry elements model extensions connected with Qt.
 *
 * Typically you should call geomExt() function with pointer to your geometry element object as argument.
 * This function return Element which provide methods which operates on your element, knowing its type.
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
 * Wrapper over plask::GeometryElement, which:
 * - has extra method used by Qt GUI,
 * - has extra data (like element name)
 */
struct ElementWrapper {

    plask::shared_ptr<plask::GeometryElement> plaskElement;

    std::string name;

    ElementWrapper(plask::shared_ptr<plask::GeometryElement> plaskElement)
        : plaskElement(plaskElement) {}

    /// Empty, virtual destructor.
    virtual ~ElementWrapper();

    /**
     * Draw geometry element using given Qt @p painter.
     * @param painter where draw element
     */
    virtual void draw(QPainter& painter) const;

    /**
     * Draw miniature with size close to given.
     * @param painter where miniature shpuld be drawn
     * @param w, h requested miniature size
     */
    virtual void drawMiniature(QPainter& painter, qreal w, qreal h) const;

    /**
     * Get miniature image with size close to given.
     * @param w, h requested miniature size
     * @return miniature
     */
    QPixmap getMiniature(qreal w, qreal h) const;

    /**
     * Get string representation of given element.
     * @return string representation of wrapped element, can have multiple lines of text
     */
    virtual QString toStr() const;

    /**
     * Fill property browser with properties of wrapped element.
     */
    virtual void setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

    void setupPropertiesBrowser(BrowserWithManagers& managers) const {
        setupPropertiesBrowser(managers, managers.browser);
    }

    /**
     * Fill property browser with properties of @p container child.
     *
     * This is called only for containers and default implementation call setupPropertiesBrowser for pointed child.
     * Typically, you can call ElementExtensionImplBase::setupPropertiesBrowserForChild in subclasses.
     * @param index real child index
     */
    virtual void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const;

    void setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers) const {
        setupPropertiesBrowserForChild(index, managers, managers.browser);
    }

};

template <typename WrappedType, typename BaseClass = ElementWrapper>
struct ElementWrapperFor: public BaseClass {

    ElementWrapperFor(plask::shared_ptr<WrappedType> plaskElement): ElementWrapper(plaskElement) {}

    WrappedType& c() const { return static_cast<WrappedType&>(*this->plaskElement); }

};

#endif // GUI_GEOMETRY_WRAPPER_ELEMENT_H
