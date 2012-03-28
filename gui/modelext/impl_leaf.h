#ifndef PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H
#define PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H

/** @file
 * This file includes implementation of geometry elements model extensions for leafs. Do not include it directly (see map.h).
 */

#include <plask/geometry/leaf.h>
#include "../utils/propbrowser.h"
#include "converter.h"

template <int dim>
QString printBlock(const plask::Block<dim>& toPrint) {
    return QString(QObject::tr("block%1d\nsize: %2"))
            .arg(dim).arg(QString(boost::lexical_cast<std::string>(toPrint.size).c_str()));
}

template <>
struct ExtImplFor< plask::Block<2> >: public ElementExtensionImplBaseFor< plask::Block<2> > {

    QString toStr(const plask::GeometryElement& el) const { return printBlock(c(el)); }

    void setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
        QtProperty *size = managers.sizeF.addProperty("size");
        managers.sizeF.setValue(size, toQtSize(c(el).size));
        dst.addProperty(size);
    }

};

template <>
struct ExtImplFor< plask::Block<3> >: public ElementExtensionImplBaseFor< plask::Block<3> > {

    QString toStr(const plask::GeometryElement& el) const { return printBlock(c(el)); }

};

#endif // PLASK_GUI_MODEL_EXT_MAP_IMPL_LEAF_H
