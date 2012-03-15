#ifndef PLASK_GUI_MODEL_EXT_TEXT_H
#define PLASK_GUI_MODEL_EXT_TEXT_H

#include <plask/geometry/element.h>
#include <QString>

void initElementsPrinters();

QString toStr(plask::GeometryElement::Type type);

QString toStr(const plask::GeometryElement& el);

#endif // PLASK_GUI_MODEL_EXT_TEXT_H
