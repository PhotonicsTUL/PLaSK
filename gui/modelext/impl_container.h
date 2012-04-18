#ifndef PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H
#define PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H

/** @file
 * This file includes implementation of geometry elements model extensions for containers. Do not include it directly (see map.h).
 */

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>
#include "../utils/propbrowser.h"

#include <QMessageBox>

template <int dim>
QString printStack(const plask::StackContainer<dim>& toPrint) {
    return QString(QObject::tr("stack%1d\n%2 children"))
            .arg(dim).arg(toPrint.getChildrenCount());
}

void setupAlignerEditor(plask::StackContainer<2>& s, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    QtProperty *align = managers.aligner.addProperty("align");
    managers.aligner.setValue(align, QString(s.getAlignerAt(index).str().c_str()));
    dst.addProperty(align);
    managers.connectString(align, [index, &s](const QString& v) {
        try {
           s.setAlignerAt(index, *plask::align::fromStrUnique<plask::align::DIRECTION_TRAN>(v.toStdString()));
        } catch (std::exception& e) {
           //QMessageBox::critical();
        }
    });
}

void setupAlignerEditor(plask::StackContainer<3>& s, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
}

template <int dim>
struct ExtImplFor< plask::StackContainer<dim> >: public ElementExtensionImplBaseFor< plask::StackContainer<dim> > {

    QString toStr(const plask::GeometryElement& el) const { return printStack(this->c(el)); }

    void setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
        QtProperty *from = managers.doubl.addProperty("from");
        managers.doubl.setValue(from, this->c(el).getBaseHeight());
        dst.addProperty(from);
        managers.connectDouble(from, [&](double v) { this->c(el).setBaseHeight(v); });
    }

    void setupPropertiesBrowserForChild(plask::GeometryElement& container, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
        setupAlignerEditor(this->c(container), index, managers, dst);
        ElementExtensionImplBase::setupPropertiesBrowserForChild(container, index, managers, dst);
    }

};

/*template <>
struct ExtImplFor< plask::StackContainer<3> >: public ElementExtensionImplBaseFor< plask::StackContainer<3> > {

    QString toStr(const plask::GeometryElement& el) const { return printStack(c(el)); }

};*/


template <int dim>
QString printMultiStack(const plask::MultiStackContainer<dim>& toPrint) {
    return QString(QObject::tr("multi-stack%1d\n%2 children (%3 repeated %4 times)"))
            .arg(dim).arg(toPrint.getChildrenCount()).arg(toPrint.getRealChildrenCount()).arg(toPrint.repeat_count);
};

template <int dim>
struct ExtImplFor< plask::MultiStackContainer<dim> >: public ElementExtensionImplBaseFor< plask::MultiStackContainer<dim> > {

    QString toStr(const plask::GeometryElement& el) const { return printMultiStack(this->c(el)); }

    void setupPropertiesBrowser(plask::GeometryElement& el, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
        // all stack properties:
        ExtImplFor< plask::StackContainer<dim> >().setupPropertiesBrowser(el, managers, dst);
        // multiple stack extras:
        QtProperty *repeat = managers.integer.addProperty("repeat count");
        managers.integer.setValue(repeat, this->c(el).repeat_count);
        managers.integer.setMinimum(repeat, 1);
        dst.addProperty(repeat);
        managers.connectInt(repeat, [&](int v) { this->c(el).setRepeatCount(v); });
    }

    void setupPropertiesBrowserForChild(plask::GeometryElement& container, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
        ExtImplFor< plask::StackContainer<dim> >().setupPropertiesBrowserForChild(container, index, managers, dst);
    }

};

/*template <>
struct ExtImplFor< plask::MultiStackContainer<3> >: public ElementExtensionImplBaseFor< plask::MultiStackContainer<3> > {

    QString toStr(const plask::GeometryElement& el) const { return printMultiStack(c(el)); }

};*/

#endif // PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H
