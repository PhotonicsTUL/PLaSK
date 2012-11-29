#include "text.h"

#include <QObject>

QString toStr(plask::GeometryObject::Type type) {
    switch (type) {
    case plask::GeometryObject::TYPE_CONTAINER: return QObject::tr("container");
    case plask::GeometryObject::TYPE_LEAF: return QObject::tr("leaf");
    case plask::GeometryObject::TYPE_SPACE_CHANGER: return QObject::tr("space changer");
    case plask::GeometryObject::TYPE_TRANSFORM: return QObject::tr("transform");
    case plask::GeometryObject::TYPE_GEOMETRY: return QObject::tr("geometry");
    case plask::GeometryObject::TYPE_SEPARATOR: return QObject::tr("separator");
    }
    return QString();
}
