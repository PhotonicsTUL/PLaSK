#include "container.h"


#include "../utils/propbrowser.h"

#include <QMessageBox>

// ------------------- StackWrapper --------------------------------

template <int dim>
QString StackWrapper<dim>::toStr() const {
    plask::GeometryElement& el = *this->wrappedElement;
    return QString(QObject::tr("stack%1d%2\n%3 children"))
        .arg(dim)
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
        .arg(el.getChildrenCount());
}

template <int dim>
void StackWrapper<dim>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    ElementWrapperFor< plask::StackContainer<dim> >::setupPropertiesBrowser(managers, dst);
    QtProperty *from = managers.doubl.addProperty("from");
    managers.doubl.setValue(from, this->c().getBaseHeight());
    dst.addProperty(from);
    managers.connectDouble(from, [&](double v) { this->c().setBaseHeight(v); });
}

static void setupAlignerEditor(plask::StackContainer<2>& s, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
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

static void setupAlignerEditor(plask::StackContainer<3>& s, std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
}

template <int dim>
void StackWrapper<dim>::setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    setupAlignerEditor(this->c(), index, managers, dst);
    ElementWrapper::setupPropertiesBrowserForChild(index, managers, dst);
}

template class StackWrapper<2>;
template class StackWrapper<3>;

// ------------------- MultiStackWrapper --------------------------

template <int dim>
QString MultiStackWrapper<dim>::toStr() const {
    auto& el = this->c();
    return QString(QObject::tr("multistack%1d%2\n%3 children (%4 repeated %5 times)"))
        .arg(dim)
        .arg(this->name.isEmpty() ? "" : (" \"" + this->name + "\""))
        .arg(el.getChildrenCount())
        .arg(el.getRealChildrenCount())
        .arg(el.repeat_count);
}

template <int dim>
void MultiStackWrapper<dim>::setupPropertiesBrowser(BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) {
    // all stack properties:
    StackWrapper<dim>::setupPropertiesBrowser(managers, dst);
    // multiple stack extras:
    QtProperty *repeat = managers.integer.addProperty("repeat count");
    managers.integer.setValue(repeat, this->c().repeat_count);
    managers.integer.setMinimum(repeat, 1);
    dst.addProperty(repeat);
    managers.connectInt(repeat, [&](int v) { this->c().setRepeatCount(v); });
}

/*template <int dim>
void MultiStackWrapper<dim>::setupPropertiesBrowserForChild(std::size_t index, BrowserWithManagers& managers, QtAbstractPropertyBrowser& dst) const {
    StackWrapper<dim>::setupPropertiesBrowserForChild(index, managers, dst);
}*/

template class MultiStackWrapper<2>;
template class MultiStackWrapper<3>;
