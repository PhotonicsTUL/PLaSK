#ifndef GUI_GEOMETRY_WRAPPER_REGISTER_H
#define GUI_GEOMETRY_WRAPPER_REGISTER_H

#include "object.h"

/**
 * Get extension for geometry object @p el.
 * @param el geometry object
 * @return extension for real type of @p el
 */
plask::shared_ptr<ObjectWrapper> ext(plask::shared_ptr<plask::GeometryObject> el);

plask::shared_ptr<ObjectWrapper> ext(const plask::GeometryObject& el);

/// Implementation of plask::GeometryObject::WriteXMLCallback which read all data from objects extensions
struct NamesFromExtensions: public plask::GeometryObject::WriteXMLCallback {

    std::string getName(const plask::GeometryObject &object, plask::AxisNames &axesNames) const;

    std::vector<std::string> getPathNames(const plask::GeometryObject &parent, const plask::GeometryObject &child, std::size_t index_of_child_in_parent) const;
};

#endif // REGISTER_H
