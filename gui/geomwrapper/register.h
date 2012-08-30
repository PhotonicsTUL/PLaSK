#ifndef GUI_GEOMETRY_WRAPPER_REGISTER_H
#define GUI_GEOMETRY_WRAPPER_REGISTER_H

#include "element.h"

/**
 * Get extension for geometry element @p el.
 * @param el geometry element
 * @return extension for real type of @p el
 */
plask::shared_ptr<ElementWrapper> ext(plask::shared_ptr<plask::GeometryElement> el);

plask::shared_ptr<ElementWrapper> ext(const plask::GeometryElement& el);

/// Implementation of plask::GeometryElement::WriteXMLCallback which read all data from elements extensions
struct NamesFromExtensions: public plask::GeometryElement::WriteXMLCallback {

    std::string getName(const plask::GeometryElement &element, plask::AxisNames &axesNames) const;

    std::vector<std::string> getPathNames(const plask::GeometryElement &parent, const plask::GeometryElement &child, std::size_t index_of_child_in_parent) const;
};

#endif // REGISTER_H
