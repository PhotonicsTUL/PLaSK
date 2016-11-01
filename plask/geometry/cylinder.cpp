#include "cylinder.h"
#include "reader.h"
#include "../manager.h"

#define PLASK_CYLINDER_NAME "cylinder"

namespace plask {

const char* Cylinder::NAME = PLASK_CYLINDER_NAME;

Cylinder::Cylinder(double radius, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material), radius(std::max(radius, 0.)), height(std::max(height, 0.))
{}

Cylinder::Cylinder(double radius, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom), radius(std::max(radius, 0.)), height(std::max(height, 0.))
{}

Cylinder::Cylinder(const Cylinder& src): GeometryObjectLeaf<3>(src), radius(src.radius), height(src.height) {}

Cylinder::Box Cylinder::getBoundingBox() const {
    return Box(vec(-radius, -radius, 0.0), vec(radius, radius, height));
}

bool Cylinder::contains(const Cylinder::DVec &p) const {
    return 0.0 <= p.vert() && p.vert() <= height &&
           std::fma(p.lon(), p.lon(), p.tran() * p.tran()) <= radius * radius;
}

void Cylinder::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    materialProvider->writeXML(dest_xml_object, axes)
            .attr("radius", radius).attr("height", height);
}

bool Cylinder::isUniform(Primitive<3>::Direction direction) const {
    return direction == Primitive<3>::DIRECTION_VERT && materialProvider->isUniform(direction);
}

shared_ptr<GeometryObject> read_cylinder(GeometryReader& reader) {
    shared_ptr<Cylinder> result(new Cylinder(
                                    reader.manager.draft ? reader.source.getAttribute("radius", 0.0) : reader.source.requireAttribute<double>("radius"),
                                    reader.manager.draft ? reader.source.getAttribute("height", 0.0) : reader.source.requireAttribute<double>("height")
                         ));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(PLASK_CYLINDER_NAME, read_cylinder);

}   // namespace plask
