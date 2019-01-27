#include "extruded_triangular3d.h"

namespace plask {

Vec<3, double> ExtrudedTriangularMesh3D::at(std::size_t index) const {
    if (vertFastest) {
        Vec<2, double> longTran = longTranMesh[index / vertAxis->size()];
        return vec(longTran.c0, longTran.c1, vertAxis->at(index % vertAxis->size()));
    } else {
        Vec<2, double> longTran = longTranMesh[index % longTranMesh.size()];
        return vec(longTran.c0, longTran.c1, vertAxis->at(index / longTranMesh.size()));
    }
}

std::size_t ExtrudedTriangularMesh3D::size() const {
    return longTranMesh.size() * vertAxis->size();
}

bool ExtrudedTriangularMesh3D::empty() const {
    return longTranMesh.empty() || vertAxis->empty();
}

void ExtrudedTriangularMesh3D::writeXML(XMLElement &object) const
{
    object.attr("type", "extruded_triangular2d");
    { auto a = object.addTag("vert"); vertAxis->writeXML(a); }
    { auto a = object.addTag("long_tran"); longTranMesh.writeXML(a); }
}

}   // namespace plask
