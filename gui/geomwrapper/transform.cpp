#include "transform.h"

#include <plask/geometry/transform.h>
#include "register.h"

#include "../modelext/text.h"

template <int dim>
QString TranslationWrapper<dim>::toStr() const {
    auto& el = this->c();
    return QString(QObject::tr("translation%1d%2\n%3"))
        .arg(dim)
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
        .arg(::toStr(el.translation));
}

template <>
void TranslationWrapper<2>::draw(QPainter& painter) const {
    QTransform transformBackup = painter.transform();
    const plask::Translation<2>& t = c();
    if (!t.hasChild()) return;
    painter.translate(t.translation.tran(), t.translation.up());
    ext(t.getChild())->draw(painter);
    painter.setTransform(transformBackup);
}

template <>
void TranslationWrapper<3>::draw(QPainter& painter) const {
    //TODO 3d
}

template <int dim>
bool TranslationWrapper<dim>::canInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) const {
    return index == 0 &&
           !this->c().hasChild() &&
            this->c().canHasAsChild(*to_insert);
}

template <int dim>
bool TranslationWrapper<dim>::tryInsert(plask::shared_ptr<plask::GeometryElement> to_insert, std::size_t index) {
    if (!this->canInsert(to_insert, index)) return false;
    this->c().setChild(plask::static_pointer_cast< plask::GeometryElementD<dim> >(to_insert));
    return true;
}

template struct TranslationWrapper<2>;
template struct TranslationWrapper<3>;

QString ExtrusionWrapper::toStr() const {
    auto& el = this->c();
    return QString(QObject::tr("extrusion %1\nlength: %2"))
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
            .arg(el.length);
}

void ExtrusionWrapper::setupPropertiesBrowser(BrowserWithManagers &managers, QtAbstractPropertyBrowser &dst) {
    ElementWrapperFor< plask::Extrusion >::setupPropertiesBrowser(managers, dst);

    QtProperty *length = managers.doubl.addProperty("length");
    managers.doubl.setValue(length, this->c().length);
    dst.addProperty(length);
    managers.connectDouble(length, [&](const double &v) { this->c().setLength(v); });
}
