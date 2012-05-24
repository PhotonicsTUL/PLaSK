#include "leaf.h"

#include <plask/geometry/leaf.h>
#include "../utils/propbrowser.h"
#include "../modelext/converter.h"

template <int dim>
QString BlockWrapper<dim>::toStr() const {
    auto& el = this->c();
    return QString(QObject::tr("block%1d%2\nsize: %3"))
        .arg(dim)
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
        .arg(QString(boost::lexical_cast<std::string>(el.size).c_str()));
}

template <>
void BlockWrapper<2>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
    QtProperty *size = managers.sizeF.addProperty("size");
    managers.sizeF.setValue(size, toQtSize(this->c().size));
    dst.addProperty(size);
    managers.connectSizeF(size, [&](const QSizeF &v) { this->c().setSize(v.width(), v.height()); });
}


template <>
void BlockWrapper<3>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
    //TODO 3d
}

template class BlockWrapper<2>;
template class BlockWrapper<3>;
