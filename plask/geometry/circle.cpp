#include "circle.h"
#include "reader.h"
#include "../manager.h"

#define PLASK_CIRCLE2D_NAME "circle"
#define PLASK_CIRCLE3D_NAME "sphere"

namespace plask {

template <int dim>
const char* Circle<dim>::NAME = dim == 2 ? PLASK_CIRCLE2D_NAME : PLASK_CIRCLE3D_NAME;

template <int dim>
std::string Circle<dim>::getTypeName() const {
    return NAME;
}

template <int dim>
Circle<dim>::Circle(double radius, const shared_ptr<plask::Material> &material)
: GeometryObjectLeaf<dim>(material), radius(radius)
{
    if (radius < 0.) radius = 0.;
}

template <>
typename Circle<2>::Box Circle<2>::getBoundingBox() const {
    return Circle<2>::Box(vec(-radius, -radius), vec(radius, radius));
}

template <>
typename Circle<3>::Box Circle<3>::getBoundingBox() const {
    return Circle<3>::Box(vec(-radius, -radius, -radius), vec(radius, radius, radius));
}

template <int dim>
bool Circle<dim>::contains(const typename Circle<dim>::DVec &p) const {
    return abs2(p) <= radius * radius;
}

template <int dim>
void Circle<dim>::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    GeometryObjectLeaf<dim>::writeXMLAttr(dest_xml_object, axes);
    this->materialProvider->writeXML(dest_xml_object, axes).attr("radius", this->radius);
}

template <int dim>
bool Circle<dim>::isUniform(plask::Primitive<3>::Direction /*direction*/) const {
    return false;
}

template <int dim>
shared_ptr<GeometryObject> read_circle(GeometryReader& reader) {
    shared_ptr< Circle<dim> > circle = plask::make_shared<Circle<dim>>(
        reader.manager.draft ? reader.source.getAttribute("radius", 0.0) : reader.source.requireAttribute<double>("radius"));
    circle->readMaterial(reader);
    reader.source.requireTagEnd();
    return circle;
}

template struct PLASK_API Circle<2>;
template struct PLASK_API Circle<3>;

static GeometryReader::RegisterObjectReader circle_reader(PLASK_CIRCLE2D_NAME, read_circle<2>);
static GeometryReader::RegisterObjectReader sphere_reader(PLASK_CIRCLE3D_NAME, read_circle<3>);

}   // namespace plask
