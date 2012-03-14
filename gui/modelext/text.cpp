#include "text.h"

#include <QObject>

QString toStr(plask::GeometryElement::Type type) {
    switch (type) {
    case plask::GeometryElement::TYPE_CONTAINER: return QObject::tr("container");
    case plask::GeometryElement::TYPE_LEAF: return QObject::tr("leaf");
    case plask::GeometryElement::TYPE_SPACE_CHANGER: return QObject::tr("space changer");
    case plask::GeometryElement::TYPE_TRANSFORM: return QObject::tr("transform");
    }
}

QString universalPrinter(plask::shared_ptr<plask::GeometryElement> el) {
    if (!el) return "";
    return QString(QObject::tr("%1%2d (%3 children)")
            .arg(toStr(el->getType())))
            .arg(el->getDimensionsCount())
            .arg(el->getChildCount());
}

QString toStr(plask::shared_ptr<plask::GeometryElement> el) {
    //TODO
    return universalPrinter(el);
}
