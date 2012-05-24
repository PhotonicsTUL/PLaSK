#ifndef GUI_GEOMETRY_WRAPPER_REGISTER_H
#define GUI_GEOMETRY_WRAPPER_REGISTER_H

#include "element.h"

/**
 * Get extension for geometry element @p el.
 * @param el geometry element
 * @return extension for real type of @p el
 */
plask::shared_ptr<ElementWrapper> ext(plask::shared_ptr<plask::GeometryElement> el);

#endif // REGISTER_H
