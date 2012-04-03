#include "map.h"

#include <unordered_map>
#include <typeinfo>
#include <typeindex>
#include <algorithm>


#include "converter.h"
#include "text.h"
#include "../utils/draw.h"

/**
 * Helper class used in impl_*.h files. Base for ElementExtensionImplBase implementations with some casting methods.
 */
template <typename ElementType>
struct ElementExtensionImplBaseFor: public ElementExtensionImplBase {

    static const ElementType& c(const plask::GeometryElement& el) { return static_cast<const ElementType&>(el); }

    static ElementType& c(plask::GeometryElement& el) { return static_cast<ElementType&>(el); }

};

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
        auto bb = toQt(static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox());
        painter.fillRect(bb, QColor(150, 100, 100));
        painter.drawRect(bb);
    } else {
        for (std::size_t i = 0; i < toDraw.getChildrenCount(); ++i)
            ext(*toDraw.getChildAt(i)).draw(painter);
    }
}

void ElementExtensionImplBase::drawMiniature(const plask::GeometryElement& toDraw, QPainter& painter, qreal w, qreal h) const {
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment

    QTransform transformBackup = painter.transform();

    painter.setTransform(flipVertical);
    painter.translate(0.0, -h);

    plask::Box2d bb = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox();

    plask::Vec<2, double> s = bb.size();
    double scale = std::min(w / s.tran, h / s.up);
    painter.scale(scale, scale);

    painter.translate(-bb.lower.tran, -bb.lower.up);

    draw(toDraw, painter);

    painter.setTransform(transformBackup);
}

QPixmap ElementExtensionImplBase::getMiniature(const plask::GeometryElement& toDraw, qreal w, qreal h) const {
    if (toDraw.getDimensionsCount() != 2)
        return QPixmap(); //we draw 2d only at this moment

    //TODO do not calc. bb. two times
    plask::Vec<2, double> s = static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox().size();
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
    drawMiniature(toDraw, painter, w-1.0, h-1.0);   //-1.0 for typical pen size
    return result;
}

QString ElementExtensionImplBase::toStr(const plask::GeometryElement& el) const {
    return QString(QObject::tr("%1%2d\n%3 children")
        .arg(::toStr(el.getType())))
        .arg(el.getDimensionsCount())
        .arg(el.getChildrenCount());
}

void ElementExtensionImplBase::setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
}

void ElementExtensionImplBase::setupPropertiesBrowser(plask::GeometryElement& container, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
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
