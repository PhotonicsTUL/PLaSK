#ifndef GUI_GEOMETRY_WRAPPER_ELEMENT_H
#define GUI_GEOMETRY_WRAPPER_ELEMENT_H

#include <plask/geometry/element.h>

struct Element {

    plask::shared_ptr<plask::GeometryElement> plaskElement;

    std::string name;

    Element(plask::shared_ptr<plask::GeometryElement> plaskElement)
        : plaskElement(plaskElement) {}



};

#endif // GUI_GEOMETRY_WRAPPER_ELEMENT_H
