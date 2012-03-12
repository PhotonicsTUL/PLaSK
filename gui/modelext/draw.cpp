#include "draw.h"

#include "converter.h"

#include <unordered_map>
#include <typeinfo>
#include <typeindex>

#include <plask/geometry/transform.h>

typedef void draw_element_f(const plask::GeometryElement& toDraw, QPainter& painter);

std::unordered_map<std::type_index, draw_element_f*> drawers;

void universalDrawer(const plask::GeometryElement& toDraw, QPainter& painter) {
    if (toDraw.isLeaf()) {
        painter.fillRect(toQt(static_cast< const plask::GeometryElementD<2>& >(toDraw).getBoundingBox()), QColor(150, 100, 100));
    } else {
        for (std::size_t i = 0; i < toDraw.getChildCount(); ++i)
            drawElement(*toDraw.getChildAt(i), painter);
    }
}

void drawElement(const plask::GeometryElement& toDraw, QPainter& painter) {
    if (toDraw.getDimensionsCount() != 2)
        return; //we draw 2d only at this moment
    auto drawer = drawers.find(std::type_index(typeid(toDraw)));
    if (drawer != drawers.end())
        drawer->second(toDraw, painter);
    else
        universalDrawer(toDraw, painter);
}

void drawTranslation(const plask::GeometryElement& toDraw, QPainter& painter) {
    QTransform transformBackup = painter.transform();
    const plask::Translation<2>& t = static_cast< const plask::Translation<2>& >(toDraw);
    if (!t.hasChild()) return;
    painter.translate(t.translation.tran, t.translation.up);
    drawElement(*t.getChild(), painter);
    painter.setTransform(transformBackup);
}

void initElementsDrawers() {
    plask::shared_ptr<plask::Translation<2>> tr2 = plask::make_shared<plask::Translation<2>>();
    drawers[std::type_index(typeid(*tr2))] = drawTranslation;
}
