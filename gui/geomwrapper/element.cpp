#include "element.h"

#include <plask/geometry/space.h>

#include "../modelext/converter.h"
#include "../modelext/text.h"
#include "../utils/draw.h"

#include "register.h"

ElementWrapper::~ElementWrapper() {
    if (this->wrappedElement) this->wrappedElement->changedDisconnectMethod(this, &ElementWrapper::onWrappedChange);
}

void ElementWrapper::draw(QPainter& painter) const {
    plask::GeometryElement& toDraw = *wrappedElement;
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment
    if (toDraw.isLeaf()) {
        auto bb = toQt(static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox());
        painter.fillRect(bb, QColor(150, 100, 100));
        //painter.setPen(QPen(QColor(0,0,0), 0.0));
        painter.drawRect(bb);
    } else {
        for (std::size_t i = 0; i < toDraw.getChildrenCount(); ++i)
            ext(toDraw.getChildAt(i))->draw(painter);
    }
}

void ElementWrapper::drawMiniature(QPainter& painter, qreal w, qreal h) const {
    plask::GeometryElement& toDraw = *wrappedElement;

    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment

    QTransform transformBackup = painter.transform();

    painter.setTransform(flipVertical);
    painter.translate(0.0, -h);

    plask::Box2D bb = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox();

    plask::Vec<2, double> s = bb.size();
    double scale = std::min(w / s.tran, h / s.up);
    painter.scale(scale, scale);

    painter.translate(-bb.lower.tran, -bb.lower.up);

    draw(painter);

    painter.setTransform(transformBackup);
}

void ElementWrapper::drawReal(QPainter &painter) const
{
    plask::GeometryElement& toDraw = *wrappedElement;
    if (toDraw.isContainer()) {
        for (std::size_t i = 0; i < toDraw.getRealChildrenCount(); ++i)
            ext(toDraw.getRealChildAt(i))->draw(painter);
    } else
        draw(painter);
}

QPixmap ElementWrapper::getMiniature(qreal w, qreal h) const {
    plask::GeometryElement& toDraw = *wrappedElement;

    if (toDraw.getDimensionsCount() != 2)
        return QPixmap(); //we draw 2d only at this moment

    //TODO do not calc. bb. two times
    plask::Vec<2, double> s;
    if (toDraw.isGeometry())
        s = static_cast< const plask::GeometryD<2>& >(toDraw).getChildBoundingBox().size();
    else
        s = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox().size();
    //plask::Vec<2, double> s = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox().size();

    double obj_prop = s.tran / s.up;
    if (obj_prop > w / h) { //obj. to wide
        h = w / obj_prop;
    } else  //obj to high
        w = h * obj_prop;

    QPixmap result(w+1, h+1);
    result.fill(QColor(255, 255, 255, 0));
    if (w < 1.0 || h < 1.0) return result;  //to small miniature
    QPainter painter;
    painter.begin(&result);           // paint in picture
    drawMiniature(painter, w-1.0, h-1.0);   //-1.0 for typical pen size
    return result;
}

QString ElementWrapper::toStr() const {
    plask::GeometryElement& el = *wrappedElement;
    return QString(QObject::tr("%1%2d%3\n%4 children")
        .arg(::toStr(el.getType())))
        .arg(el.getDimensionsCount())
        .arg(name.isEmpty() ? "" : (" \"" + name + "\""))
        .arg(el.getChildrenCount());
}

void ElementWrapper::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    QtProperty *nameProp = managers.string.addProperty("name");
    managers.string.setValue(nameProp, getName());
    dst.addProperty(nameProp);
    managers.connectString(nameProp, [this](const QString& v) {
        this->setName(v);
    });
}

void ElementWrapper::setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    plask::shared_ptr<plask::GeometryElement> e = wrappedElement->getRealChildAt(index);
    if (e->getRealChildrenCount() == 0) return;
    ext(e->getRealChildAt(0))->setupPropertiesBrowser(managers, dst);
}

/*QPixmap drawMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h) {
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment
    auto bb = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox();
}*/
