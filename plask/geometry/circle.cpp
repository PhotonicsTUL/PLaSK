#include "circle.h"
#include "../manager.h"
#include "reader.h"

#define PLASK_CIRCLE2D_NAME "circle"
#define PLASK_CIRCLE3D_NAME "sphere"

namespace plask {

template <int dim> const char* Circle<dim>::NAME = dim == 2 ? PLASK_CIRCLE2D_NAME : PLASK_CIRCLE3D_NAME;

template <int dim> std::string Circle<dim>::getTypeName() const { return NAME; }

template <int dim>
Circle<dim>::Circle(double radius, const shared_ptr<plask::Material>& material)
    : GeometryObjectLeaf<dim>(material), radius(radius) {
    if (radius < 0.) radius = 0.;
}

template <> typename Circle<2>::Box Circle<2>::getBoundingBox() const {
    return Circle<2>::Box(vec(-radius, -radius), vec(radius, radius));
}

template <> typename Circle<3>::Box Circle<3>::getBoundingBox() const {
    return Circle<3>::Box(vec(-radius, -radius, -radius), vec(radius, radius, radius));
}

template <int dim> bool Circle<dim>::contains(const typename Circle<dim>::DVec& p) const {
    return abs2(p) <= radius * radius;
}

template <int dim> void Circle<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<dim>::writeXMLAttr(dest_xml_object, axes);
    this->materialProvider->writeXML(dest_xml_object, axes).attr("radius", this->radius);
}

template <int dim>
void Circle<dim>::addPointsAlong(std::set<double>& points,
                                Primitive<3>::Direction direction,
                                unsigned max_steps,
                                double min_step_size) const {
    assert(int(direction) >= 3 - dim && int(direction) <= 3);
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    unsigned steps = min(unsigned(2. * radius / min_step_size), max_steps);
    double step = 2. * radius / steps;
    for (unsigned i = 0; i <= steps; ++i) points.insert(i * step - radius);
}

template <>
void Circle<2>::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const {
    // TODO
}

template <>
void Circle<3>::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const {
    // TODO
}

template <int dim> shared_ptr<GeometryObject> read_circle(GeometryReader& reader) {
    shared_ptr<Circle<dim>> circle =
        plask::make_shared<Circle<dim>>(reader.manager.draft ? reader.source.getAttribute("radius", 0.0)
                                                             : reader.source.requireAttribute<double>("radius"));
    circle->readMaterial(reader);
    reader.source.requireTagEnd();
    return circle;
}

template struct PLASK_API Circle<2>;
template struct PLASK_API Circle<3>;

static GeometryReader::RegisterObjectReader circle_reader(PLASK_CIRCLE2D_NAME, read_circle<2>);
static GeometryReader::RegisterObjectReader sphere_reader(PLASK_CIRCLE3D_NAME, read_circle<3>);

}  // namespace plask
