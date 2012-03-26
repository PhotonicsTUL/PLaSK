#include "draw.h"

#include "converter.h"
#include "map.h"

#include <plask/geometry/transform.h>

//-------- GeometryElementItem --------

void GeometryElementItem::doUpdate() {
    this->boundingBox = toQt(element->getBoundingBox());
    this->update();
}

void GeometryElementItem::onElementUpdate(plask::GeometryElement::Event &evt) {
    doUpdate();
}

void GeometryElementItem::setElement(const plask::shared_ptr< const plask::GeometryElementD<2> >& element) {
    this->element = element;
    doUpdate();
}

QRectF GeometryElementItem::boundingRect() const {
    return boundingBox;
}

void GeometryElementItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    ext(*element).draw(*painter);
}

int GeometryElementItem::type() const {
     return UserType + 0;
}


