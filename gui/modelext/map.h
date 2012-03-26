#ifndef PLASK_GUI_MODEL_EXT_MAP_H
#define PLASK_GUI_MODEL_EXT_MAP_H

#include <QPainter>
#include <QGraphicsItem>

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
class QRectF;
QT_END_NAMESPACE

#include <plask/geometry/element.h>

struct ElementExtensionImplBase {

    virtual ~ElementExtensionImplBase();

    virtual void draw(const plask::GeometryElement& toDraw, QPainter& painter) const;

    //QPixmap drawMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h);

    virtual QString toStr(const plask::GeometryElement& el) const;

};

template <typename ElementType>
struct ElementExtensionImplBaseFor: public ElementExtensionImplBase {

    static const ElementType& c(const plask::GeometryElement& el) { return static_cast<const ElementType&>(el); }

    static ElementType& c(plask::GeometryElement& el) { return static_cast<ElementType&>(el); }

};

struct ElementExtension {

    const ElementExtensionImplBase& impl;

    plask::GeometryElement& element;

    ElementExtension(const ElementExtensionImplBase& impl, plask::GeometryElement& element)
        : impl(impl), element(element) {}

    void draw(QPainter& painter) const { impl.draw(element, painter); }

    QString toStr() const {  return impl.toStr(element); }

};

void initModelExtensions();

ElementExtension ext(plask::GeometryElement& el);

inline ElementExtension ext(const plask::GeometryElement& el) {
    return ext(const_cast<plask::GeometryElement&>(el));
}

#endif // PLASK_GUI_MODEL_EXT_MAP_H
