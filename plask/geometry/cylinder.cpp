#include "cylinder.h"
#include "reader.h"

namespace plask {

Cylinder::Cylinder(double radius, double height, const shared_ptr<Material>& material)
    : GeometryObjectLeaf<3>(material), radius(radius), height(height)
{}

Cylinder::Box Cylinder::getBoundingBox() const {
    return Box(vec(- radius, - radius, 0.0), vec(+ radius, + radius, height));
}

bool Cylinder::contains(const Cylinder::DVec &p) const {
    return 0.0 >= p.vert() && p.vert() <= height &&
            std::fma(p.lon(), p.lon(), p.tran() * p.tran()) <= radius * radius;
}

void Cylinder::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    dest_xml_object.attr("radius", radius)
                    .attr("height", height)
                    .attr(GeometryReader::XML_MATERIAL_ATTR, material->str());
}

shared_ptr<GeometryObject> read_cylinder(GeometryReader& reader) {
    shared_ptr< Cylinder > result(new Cylinder(
                               reader.source.requireAttribute<double>("radius"),
                               reader.source.requireAttribute<double>("height"),
                               reader.requireMaterial()
                           ));
    reader.source.requireTagEnd();
    return result;
}

static GeometryReader::RegisterObjectReader cylinder_reader(Cylinder::NAME, read_cylinder);

}   // namespace plask
