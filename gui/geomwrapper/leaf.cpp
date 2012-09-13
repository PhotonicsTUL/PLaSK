#include "leaf.h"

#include <plask/geometry/leaf.h>
#include "../utils/propbrowser.h"
#include "../modelext/converter.h"

#include "../material.h"

template <int dim>
QString BlockWrapper<dim>::toStr() const {
    auto& el = this->c();
    return QString(QObject::tr("block%1d%2\nsize: %3"))
        .arg(dim)
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
        .arg(QString(boost::lexical_cast<std::string>(el.size).c_str()));
}

template <>
void BlockWrapper<2>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    ObjectWrapperFor< plask::Block<2> >::setupPropertiesBrowser(managers, dst);

    QtProperty *size = managers.sizeF.addProperty("size");
    managers.sizeF.setValue(size, toQtSize(this->c().size));
    dst.addProperty(size);
    managers.connectSizeF(size, [&](const QSizeF &v) { this->c().setSize(v.width(), v.height()); });

    QtProperty *material = managers.string.addProperty("material");
    plask::shared_ptr<NameOnlyMaterial> mat = plask::static_pointer_cast<NameOnlyMaterial>(this->c().material);
    managers.string.setValue(material, mat ? QString(mat->name().c_str()) : "");
    dst.addProperty(material);
    managers.connectString(material, [&](const QString &v) {
                           plask::shared_ptr<NameOnlyMaterial> mat = plask::static_pointer_cast<NameOnlyMaterial>(this->c().material);
                           if (v.isEmpty()) {
                                mat = plask::shared_ptr<NameOnlyMaterial>();
                           } else {
                                if (!mat) mat = NameOnlyMaterial::getInstance(v.toStdString());
                                else mat->setName(v.toStdString());
                           }
                           this->c().setMaterial(mat);
    });
}


template <>
void BlockWrapper<3>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    ObjectWrapperFor< plask::Block<3> >::setupPropertiesBrowser(managers, dst);
    //TODO 3d
}

template struct BlockWrapper<2>;
template struct BlockWrapper<3>;
