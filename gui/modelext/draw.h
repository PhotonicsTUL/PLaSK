#ifndef PLASK_GUI_MODEL_EXT_DRAW_H
#define PLASK_GUI_MODEL_EXT_DRAW_H

#include <QPainter>
#include <plask/geometry/element.h>

QT_BEGIN_NAMESPACE
class QPainter;
QT_END_NAMESPACE

void drawElement(const plask::GeometryElement& toDraw, QPainter& painter);

void initElementsDrawers();

#endif // PLASK_GUI_MODEL_EXT_DRAW_H
