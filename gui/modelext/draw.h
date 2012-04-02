#ifndef PLASK_GUI_MODEL_EXT_DRAW_H
#define PLASK_GUI_MODEL_EXT_DRAW_H

#include <plask/geometry/element.h>
#include <QGraphicsItem>

QT_BEGIN_NAMESPACE
class QPainter;
class QGraphicsItem;
QT_END_NAMESPACE

/**
 * Wrapper over plask::GeometryElementD<2> which implement QGraphicsItem interface.
 */
class GeometryElementItem: public QGraphicsItem {

    plask::weak_ptr< plask::GeometryElementD<2> > element;

    //cache:
    QRectF boundingBox;

    void doUpdate(plask::shared_ptr< const plask::GeometryElementD<2> > e, bool resized = true);

    void doUpdate(bool resized = true) { doUpdate(element.lock(), resized); }

    void onElementUpdate(const plask::GeometryElement::Event& evt);

    void disconnectOnChanged();

public:

    ~GeometryElementItem();

    void setElement(const plask::shared_ptr< plask::GeometryElementD<2> >& element);

    const plask::weak_ptr< plask::GeometryElementD<2> >& getElement() const {
        return this->element;
    }

    GeometryElementItem(const plask::shared_ptr< plask::GeometryElementD<2> >& element) {
        setElement(element);
    }

    //---- QGraphicsItem methods implementation: -----

    QRectF boundingRect() const;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);

    int type() const;

};

#endif // PLASK_GUI_MODEL_EXT_DRAW_H
