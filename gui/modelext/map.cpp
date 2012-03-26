#include "map.h"

#include <unordered_map>
#include <typeinfo>
#include <typeindex>

#include "converter.h"
#include "text.h"

template <typename plaskGeomElemType>
struct ExtImplFor: public ElementExtensionImplBase {};

/// Extensions for geometry element map: type id index -> implementation of extensions
std::unordered_map<std::type_index, ElementExtensionImplBase*> extensions;

/// Universal implementation of extensions.
ElementExtensionImplBase baseImpl;

//------------ ElementExtensionImplBase ----------------------------------------------------
ElementExtensionImplBase::~ElementExtensionImplBase() {}

void ElementExtensionImplBase::draw(const plask::GeometryElement& toDraw, QPainter& painter) const {
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment
    if (toDraw.isLeaf()) {
        painter.fillRect(toQt(static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox()), QColor(150, 100, 100));
        painter.drawRect(toQt(static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox()));
    } else {
        for (std::size_t i = 0; i < toDraw.getChildCount(); ++i)
            ext(*toDraw.getChildAt(i)).draw(painter);
    }
}

QString ElementExtensionImplBase::toStr(const plask::GeometryElement& el) const {
    return QString(QObject::tr("%1%2d\n%3 children")
        .arg(::toStr(el.getType())))
        .arg(el.getDimensionsCount())
        .arg(el.getChildCount());
}

/*QPixmap drawMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h) {
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment
    auto bb = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox();
}*/

ElementExtension ext(plask::GeometryElement& el) {
    auto extensionImpl = extensions.find(std::type_index(typeid(el)));
    return ElementExtension(extensionImpl != extensions.end() ? *extensionImpl->second : baseImpl, el);
}

#include "impl_transform.h"
#include "impl_leaf.h"
#include "impl_container.h"

template <typename T>
void appendExt() {
    static ExtImplFor<T> impl;
    plask::shared_ptr<T> o = plask::make_shared<T>();
    extensions[std::type_index(typeid(*o))] = &impl;
}

void initModelExtensions() {
    appendExt< plask::Translation<2> >();
    appendExt< plask::Translation<3> >();
    appendExt< plask::StackContainer<2> >();
    appendExt< plask::StackContainer<3> >();
    appendExt< plask::MultiStackContainer<2> >();
    appendExt< plask::MultiStackContainer<3> >();
    appendExt< plask::Block<2> >();
    appendExt< plask::Block<3> >();
}
