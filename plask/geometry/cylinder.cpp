#include "cylinder.h"
#include "reader.h"

namespace plask {

Cylinder::Cylinder(double radius, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material), radius(radius), height(height)
{
    if (radius < 0.) radius = 0.;
    if (height < 0.) height = 0.;
}

Cylinder::Cylinder(double radius, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf<3>(materialTopBottom), radius(radius), height(height)
{
    if (radius < 0.) radius = 0.;
    if (height < 0.) height = 0.;
}

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
                                    reader.source.requireAttribute<double>("radius"),
                                    reader.source.requireAttribute<double>("height")
                         ));
    result->readMaterial(reader);
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(Cylinder::NAME, read_cylinder);

}   // namespace plask
