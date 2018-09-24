#include "prism.h"
#include "reader.h"

#include "../manager.h"

#define PLASK_PRISM_NAME "prism"

namespace plask {

const char* Prism::NAME = PLASK_PRISM_NAME;

std::string Prism::getTypeName() const {
    return NAME;
}

Prism::Prism(const Prism::Vec2 &p0, const Prism::Vec2 &p1, double height, const shared_ptr<Material> &material)
    : BaseClass(material), p0(p0), p1(p1), height(height)
{}

Prism::Prism(const Prism::Vec2 &p0, const Prism::Vec2 &p1, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : BaseClass(materialTopBottom), p0(p0), p1(p1), height(height)
{}


Box3D Prism::getBoundingBox() const {
    return Box3D(
                    std::min(std::min(p0.c0, p1.c0), 0.0),
                    std::min(std::min(p0.c1, p1.c1), 0.0),
                    0.,
                    std::max(std::max(p0.c0, p1.c0), 0.0),
                    std::max(std::max(p0.c1, p1.c1), 0.0),
                    height
                );
}

inline static double sign(const Vec<3,double>& p1, const Vec<2,double>& p2, const Vec<2,double>& p3) {
  return (p1.c0 - p3.c0) * (p2.c1 - p3.c1) - (p2.c0 - p3.c0) * (p1.c1 - p3.c1);
}

// Like sign, but with p3 = (0, 0)
inline static double sign0(const Vec<3,double>& p1, const Vec<2,double>& p2) {
  return (p1.c0) * (p2.c1) - (p2.c0) * (p1.c1);
}

bool Prism::contains(const Prism::DVec &p) const {
    if (p.c2 < 0 || p.c2 > height) return false;
    // algorithm comes from:
    // http://stackoverflow.com/questions/2049582/how-to-determine-a-point-in-a-triangle
    // with: v1 -> p0, v2 -> p1, v3 -> (0, 0)
    // maybe barycentric method would be better?
    bool b1 = sign(p, p0, p1) < 0.0;
    bool b2 = sign0(p, p1) < 0.0;
    return (b1 == b2) && (b2 == (sign(p, Primitive<2>::ZERO_VEC, p0) < 0.0));
}

void Prism::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
                    .attr("a" + axes.getNameForLong(), p0.tran())
                    .attr("a" + axes.getNameForTran(), p0.vert())
                    .attr("b" + axes.getNameForLong(), p1.tran())
                    .attr("b" + axes.getNameForTran(), p1.vert())
                    .attr("height", height);
}

bool Prism::isUniform(Primitive<3>::Direction direction) const {
    return direction == Primitive<3>::DIRECTION_VERT;
}

shared_ptr<GeometryObject> read_prism(GeometryReader& reader) {
    shared_ptr<Prism> prism(new Prism());
    if (reader.manager.draft) {
        prism->p0.c0 = reader.source.getAttribute("a" + reader.getAxisLongName(), 0.0);
        prism->p0.c1 = reader.source.getAttribute("a" + reader.getAxisTranName(), 0.0);
        prism->p1.c0 = reader.source.getAttribute("b" + reader.getAxisLongName(), 0.0);
        prism->p1.c1 = reader.source.getAttribute("b" + reader.getAxisTranName(), 0.0);
        prism->height = reader.source.getAttribute("height", 0.0);
    } else {
        prism->p0.c0 = reader.source.requireAttribute<double>("a" + reader.getAxisLongName());
        prism->p0.c1 = reader.source.requireAttribute<double>("a" + reader.getAxisTranName());
        prism->p1.c0 = reader.source.requireAttribute<double>("b" + reader.getAxisLongName());
        prism->p1.c1 = reader.source.requireAttribute<double>("b" + reader.getAxisTranName());
        prism->height = reader.source.requireAttribute<double>("height");
    }
    prism->readMaterial(reader);
    reader.source.requireTagEnd();
    return prism;
}

static GeometryReader::RegisterObjectReader prism_reader(PLASK_PRISM_NAME, read_prism);

}   // namespace plask
