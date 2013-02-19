#ifndef PLASK__GEOMETRY_CYLINDER_H
#define PLASK__GEOMETRY_CYLINDER_H

#include "leaf.h"

namespace plask {

/**
 * Cylinder with given height and radius of base.
 *
 * Center of cylinders' base lies in point (0.0, 0.0, 0.0)
 */
class Cylinder: public GeometryObjectLeaf<3> {

    double radius, height;

public:

    static constexpr const char* NAME = "cylinder";

    virtual std::string getTypeName() const { return NAME; }

    Cylinder(double radius, double height, const shared_ptr<Material>& material = shared_ptr<Material>());

    virtual Box getBoundingBox() const;

    virtual bool includes(const DVec& p) const;

    //virtual bool intersects(const Box& area) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

};

} // namespace plask

#endif // PLASK__GEOMETRY_CYLINDER_H
