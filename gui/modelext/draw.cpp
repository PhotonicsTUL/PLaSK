#include "draw.h"

#include "converter.h"
#include "../geomwrapper/register.h"

#include <plask/geometry/transform.h>

//-------- GeometryElementItem --------

void GeometryElementItem::doUpdate(plask::shared_ptr< const plask::GeometryElementD<2> > e, bool resized) {
    if (resized && e) {
        prepareGeometryChange();
        this->boundingBox = toQt(e->getBoundingBox());
    } else
        this->update();
}

void GeometryElementItem::onElementUpdate(const plask::GeometryElement::Event &evt) {
    if (evt.isDelete()) {
        //TODO possible delete *this ?
        disconnectOnChanged();
        element.reset();
        prepareGeometryChange();
        this->boundingBox = QRectF();
    } else
        doUpdate(evt.isResize());
}

void GeometryElementItem::disconnectOnChanged() {
    if (auto shared = this->element.lock()) {
        shared->changed.disconnect(boost::bind(&GeometryElementItem::onElementUpdate, this, _1));
    }
}

GeometryElementItem::~GeometryElementItem() {
    disconnectOnChanged();
}

void GeometryElementItem::setElement(const plask::shared_ptr< plask::GeometryElementD<2> >& element) {
    disconnectOnChanged();
    this->element = element;
    element->changed.connect(boost::bind(&GeometryElementItem::onElementUpdate, this, _1));
    doUpdate();
}

QRectF GeometryElementItem::boundingRect() const {
    return boundingBox;
}

void GeometryElementItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if (auto shared = element.lock()) {
        ext(shared)->draw(*painter);
    }
}

int GeometryElementItem::type() const {
     return UserType + 0;
}


