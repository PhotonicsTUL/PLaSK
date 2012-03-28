#ifndef PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H
#define PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H

/** @file
 * This file includes implementation of geometry elements model extensions for containers. Do not include it directly (see map.h).
 */

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>

template <int dim>
QString printStack(const plask::StackContainer<dim>& toPrint) {
    return QString(QObject::tr("stack%1d\n%2 children"))
            .arg(dim).arg(toPrint.getChildCount());
}

template <>
struct ExtImplFor< plask::StackContainer<2> >: public ElementExtensionImplBaseFor< plask::StackContainer<2> > {

    QString toStr(const plask::GeometryElement& el) const { return printStack(c(el)); }

};

template <>
struct ExtImplFor< plask::StackContainer<3> >: public ElementExtensionImplBaseFor< plask::StackContainer<3> > {

    QString toStr(const plask::GeometryElement& el) const { return printStack(c(el)); }

};


template <int dim>
QString printMultiStack(const plask::MultiStackContainer<dim>& toPrint) {
    return QString(QObject::tr("multi-stack%1d\n%2 children (%3 repeated %4 times)"))
            .arg(dim).arg(toPrint.getChildCount()).arg(toPrint.getRealChildCount()).arg(toPrint.repeat_count);
};

template <>
struct ExtImplFor< plask::MultiStackContainer<2> >: public ElementExtensionImplBaseFor< plask::MultiStackContainer<2> > {

    QString toStr(const plask::GeometryElement& el) const { return printMultiStack(c(el)); }

};

template <>
struct ExtImplFor< plask::MultiStackContainer<3> >: public ElementExtensionImplBaseFor< plask::MultiStackContainer<3> > {

    QString toStr(const plask::GeometryElement& el) const { return printMultiStack(c(el)); }

};


#endif // PLASK_GUI_MODEL_EXT_MAP_IMPL_CONTAINER_H
