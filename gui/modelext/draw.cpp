#include "draw.h"

#include "converter.h"
#include "../geomwrapper/register.h"

#include <plask/geometry/transform.h>

//-------- GeometryObjectItem --------

void GeometryObjectItem::doUpdate(plask::shared_ptr< const plask::GeometryObjectD<2> > e, bool resized) {
    if (resized && e) {
        prepareGeometryChange();
        this->boundingBox = toQt(e->getBoundingBox());
    } else
        this->update();
}

void GeometryObjectItem::onObjectUpdate(const plask::GeometryObject::Event &evt) {
    if (evt.isDelete()) {
        //TODO possible delete *this ?
        disconnectOnChanged();
        object.reset();
        prepareGeometryChange();
        this->boundingBox = QRectF();
    } else
        doUpdate(evt.isResize());
}

void GeometryObjectItem::disconnectOnChanged() {
    if (auto shared = this->object.lock()) {
        shared->changed.disconnect(boost::bind(&GeometryObjectItem::onObjectUpdate, this, _1));
    }
}

GeometryObjectItem::~GeometryObjectItem() {
    disconnectOnChanged();
}

void GeometryObjectItem::setObject(const plask::shared_ptr< plask::GeometryObjectD<2> >& object) {
    disconnectOnChanged();
    this->object = object;
    object->changed.connect(boost::bind(&GeometryObjectItem::onObjectUpdate, this, _1));
    doUpdate();
}

QRectF GeometryObjectItem::boundingRect() const {
    return boundingBox;
}

void GeometryObjectItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if (auto shared = object.lock()) {
        ext(shared)->draw(*painter);
    }
}

int GeometryObjectItem::type() const {
     return UserType + 0;
}


