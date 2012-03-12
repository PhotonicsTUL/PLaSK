#ifndef PLASK_GUI_MODEL_EXT_CONVERTER_H
#define PLASK_GUI_MODEL_EXT_CONVERTER_H

#include <QRectF>
#include <plask/geometry/primitives.h>

/** @file
This file includes convneters between qt and plask primitives.
*/

inline QRectF toQt(const plask::Box2d& box) {
    auto size = box.size();
    return QRectF(box.lower.tran, box.lower.up, size.tran, size.up);
}

#endif // PLASK_GUI_MODEL_EXT_CONVERTER_H
