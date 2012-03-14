#ifndef PLASK_GUI_MODEL_EXT_TEXT_H
#define PLASK_GUI_MODEL_EXT_TEXT_H

#include <plask/geometry/element.h>
#include <QString>

QString toStr(plask::GeometryElement::Type type);

QString toStr(plask::shared_ptr<plask::GeometryElement> el);

#endif // PLASK_GUI_MODEL_EXT_TEXT_H
