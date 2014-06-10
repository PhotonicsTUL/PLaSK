#include "basic.h"

namespace plask {

template struct OnePointMesh<2>;
template struct OnePointMesh<3>;

template <>
void OnePointMesh<3>::writeXML(XMLElement& object) const {
    object.attr("type", "point3d"); // this is required attribute for the provided object
    object.addTag("point")
           .attr("c0", point.c0)
           .attr("c1", point.c1)
           .attr("c2", point.c2);
}

template <>
void OnePointMesh<2>::writeXML(XMLElement &object) const {
    object.attr("type", "point2d"); // this is required attribute for the provided object
    object.addTag("point")
           .attr("c0", point.c0)
           .attr("c1", point.c1);
}

static shared_ptr<Mesh> readOnePoint3DMesh(XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    double c2 = reader.requireAttribute<double>("c2");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return make_shared<OnePointMesh<3>>(vec(c0, c1, c2));
}

static RegisterMeshReader onepoint3dmesh_reader("point3d", &readOnePoint3DMesh);

static shared_ptr<Mesh> readOnePoint2DMesh(XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return make_shared<OnePointMesh<2>>(vec(c0, c1));
}

static RegisterMeshReader onepoint2dmesh_reader("point2d", &readOnePoint2DMesh);

template struct TranslatedMesh<2>;
template struct TranslatedMesh<3>;

}   // namespace plask
