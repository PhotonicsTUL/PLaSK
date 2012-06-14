#ifndef PLASK_GUI_MODEL_EXT_CONVERTER_H
#define PLASK_GUI_MODEL_EXT_CONVERTER_H

#include <QRectF>
#include <QSizeF>
#include <plask/geometry/primitives.h>

QT_BEGIN_NAMESPACE
class QRectF;
QT_END_NAMESPACE

/** @file
This file includes convneters between qt and plask primitives.
*/

inline QRectF toQt(const plask::Box2d& box) {
    auto size = box.size();
    return QRectF(box.lower.tran, box.lower.up, size.tran, size.up);
}

inline QSizeF toQtSize(const plask::Vec<2, double>& plask_size) {
    return QSizeF(plask_size.tran, plask_size.up);
}

inline const plask::Box2d fromQt(const QRectF& r) {
    return plask::Box2d(r.left(), r.top(), r.right(), r.bottom());
}

inline plask::Vec<2, double> fromQt(const QPointF& p) {
    return plask::Vec<2, double>(p.x(), p.y());
}

#endif // PLASK_GUI_MODEL_EXT_CONVERTER_H
