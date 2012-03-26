#ifndef PLASK_GUI_MODEL_EXT_MAP_IMPL_TRANSFORM_H
#define PLASK_GUI_MODEL_EXT_MAP_IMPL_TRANSFORM_H

/** @file
 * This file includes implementation of geometry elements model extensions for transforms. Do not include it directly (see map.h).
 */

#include <plask/geometry/transform.h>

template <int dim>
inline QString printTranslation(const plask::Translation<dim>& toPrint) {
    return QString(QObject::tr("translation%1d %2"))
            .arg(dim).arg(toStr(toPrint.translation));
}

template <>
struct ExtImplFor< plask::Translation<2> >: public ElementExtensionImplBaseFor< plask::Translation<2> > {

    void draw(const plask::GeometryElement& toDraw, QPainter& painter) const {
        QTransform transformBackup = painter.transform();
        const plask::Translation<2>& t = c(toDraw);
        if (!t.hasChild()) return;
        painter.translate(t.translation.tran, t.translation.up);
        ext(*t.getChild()).draw(painter);
        painter.setTransform(transformBackup);
    }

    QString toStr(const plask::GeometryElement& el) const {
        return printTranslation(c(el));
    }

};

template <>
struct ExtImplFor< plask::Translation<3> >: public ElementExtensionImplBaseFor< plask::Translation<3> > {

    QString toStr(const plask::GeometryElement& el) const {
        return printTranslation(c(el));
    }

};

#endif // PLASK_GUI_MODEL_EXT_MAP_IMPL_TRANSFORM_H
