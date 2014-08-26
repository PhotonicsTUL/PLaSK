#include "triangle.h"
#include "reader.h"

namespace plask {

std::string Triangle::getTypeName() const {
    return NAME;
}

Triangle::Triangle(const Triangle::DVec &p0, const Triangle::DVec &p1, const shared_ptr<Material> &material)
    : GeometryObjectLeaf<2>(material), p0(p0), p1(p1)
{}

Triangle::Triangle(const Triangle::DVec &p0, const Triangle::DVec &p1, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
    : GeometryObjectLeaf<2>(materialTopBottom), p0(p0), p1(p1)
{}

Box2D Triangle::getBoundingBox() const {
    return Box2D(
                    std::min(std::min(p0.c0, p1.c0), 0.0),
                    std::min(std::min(p0.c1, p1.c1), 0.0),
                    std::max(std::max(p0.c0, p1.c0), 0.0),
                    std::max(std::max(p0.c1, p1.c1), 0.0)
                );
}

inline static double sign(const Vec<2, double>& p1, const Vec<2, double>& p2, const Vec<2, double>& p3) {
  return (p1.c0 - p3.c0) * (p2.c1 - p3.c1) - (p2.c0 - p3.c0) * (p1.c1 - p3.c1);
}

// Like sign, but with p3 = (0, 0)
inline static double sign0(const Vec<2, double>& p1, const Vec<2, double>& p2) {
  return (p1.c0) * (p2.c1) - (p2.c0) * (p1.c1);
}

bool Triangle::contains(const Triangle::DVec &p) const {
    // algorithm comes from:
    // http://stackoverflow.com/questions/2049582/how-to-determine-a-point-in-a-triangle
    // with: v1 -> p0, v2 -> p1, v3 -> (0, 0)
    // maybe barycentric method would be better?
    bool b1 = sign(p, p0, p1) < 0.0;
    bool b2 = sign0(p, p1) < 0.0;
    return (b1 == b2) && (b2 == (sign(p, Primitive<2>::ZERO_VEC, p0) < 0.0));
}

void Triangle::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const
{
    materialProvider->writeXML(dest_xml_object, axes)
                    .attr("a" + axes.getNameForTran(), p0.tran())
                    .attr("a" + axes.getNameForVert(), p0.vert())
                    .attr("b" + axes.getNameForTran(), p1.tran())
                    .attr("b" + axes.getNameForVert(), p1.vert());
}

shared_ptr<GeometryObject> read_triangle(GeometryReader& reader) {
    shared_ptr< Triangle > triangle(new Triangle());
    triangle->p0.tran() = reader.source.requireAttribute<double>("a" + reader.getAxisTranName());
    triangle->p0.vert() = reader.source.requireAttribute<double>("a" + reader.getAxisVertName());
    triangle->p1.tran() = reader.source.requireAttribute<double>("b" + reader.getAxisTranName());
    triangle->p1.vert() = reader.source.requireAttribute<double>("b" + reader.getAxisVertName());
    triangle->readMaterial(reader);
    reader.source.requireTagEnd();
    return triangle;
}

static GeometryReader::RegisterObjectReader triangle_reader(Triangle::NAME, read_triangle);

}   // namespace plask
