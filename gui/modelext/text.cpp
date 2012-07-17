#include "text.h"

#include <QObject>

QString toStr(plask::GeometryElement::Type type) {
    switch (type) {
    case plask::GeometryElement::TYPE_CONTAINER: return QObject::tr("container");
    case plask::GeometryElement::TYPE_LEAF: return QObject::tr("leaf");
    case plask::GeometryElement::TYPE_SPACE_CHANGER: return QObject::tr("space changer");
    case plask::GeometryElement::TYPE_TRANSFORM: return QObject::tr("transform");
    case plask::GeometryElement::TYPE_GEOMETRY: return QObject::tr("geometry");
    }
    return QString();
}
