#ifndef PLASK_GUI_MODEL_EXT_DRAW_H
#define PLASK_GUI_MODEL_EXT_DRAW_H

#include <QPainter>
#include <QGraphicsItem>

#include <plask/geometry/element.h>

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
class QRectF;
QT_END_NAMESPACE

void drawElement(const plask::GeometryElement& toDraw, QPainter& painter);

//QPixmap drawMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h);

void initElementsDrawers();

/**
 * Wrapper over plask::GeometryElementD<2> which implement QGraphicsItem interface.
 */
class GeometryElementItem: public QGraphicsItem {

    //TODO weak_ptr
    plask::shared_ptr< const plask::GeometryElementD<2> > element;

    //cache:
    QRectF boundingBox;

    void doUpdate();

    void onElementUpdate(plask::GeometryElement::Event& evt);

public:

    void setElement(const plask::shared_ptr< const plask::GeometryElementD<2> >& element);

    const plask::shared_ptr< const plask::GeometryElementD<2> >& getElement() const {
        return element;
    }

    GeometryElementItem(const plask::shared_ptr< const plask::GeometryElementD<2> >& element) {
        setElement(element);
    }

    //---- QGraphicsItem methods implementation: -----

    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    int type() const;

};

#endif // PLASK_GUI_MODEL_EXT_DRAW_H
