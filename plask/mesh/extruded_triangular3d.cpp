#include "extruded_triangular3d.h"

namespace plask {

inline Vec<3, double> from_longTran_vert(const Vec<2, double>& longTran, const double& vert) {
    return vec(longTran.c0, longTran.c1, vert);
}

ExtrudedTriangularMesh3D::Element::Element(const ExtrudedTriangularMesh3D &mesh, std::size_t elementIndex)
    : mesh(mesh)
{
    if (mesh.vertFastest) {
        const std::size_t seg_size = mesh.vertAxis->size()-1;
        longTranIndex = elementIndex / seg_size;
        vertIndex = elementIndex % seg_size;
    } else {
        const std::size_t seg_size = mesh.longTranMesh.getElementsCount();
        longTranIndex = elementIndex % seg_size;
        vertIndex = elementIndex / seg_size;
    }
}

Vec<3, double> ExtrudedTriangularMesh3D::Element::getMidpoint() const {
    return from_longTran_vert(
        longTranElement().getMidpoint(),
        (mesh.vertAxis->at(vertIndex) + mesh.vertAxis->at(vertIndex+1)) * 0.5
    );
}

double ExtrudedTriangularMesh3D::Element::getArea() const {
    return longTranElement().getArea() *
            (mesh.vertAxis->at(vertIndex+1) - mesh.vertAxis->at(vertIndex));
}

Box3D ExtrudedTriangularMesh3D::Element::getBoundingBox() const {
    Box2D ltBox = longTranElement().getBoundingBox();
    return Box3D(
              from_longTran_vert(ltBox.lower, mesh.vertAxis->at(vertIndex)),
              from_longTran_vert(ltBox.upper, mesh.vertAxis->at(vertIndex+1))
           );
}

Vec<3, double> ExtrudedTriangularMesh3D::at(std::size_t index) const {
    if (vertFastest) {
        const std::size_t seg_size = vertAxis->size();
        return from_longTran_vert(longTranMesh[index / seg_size], vertAxis->at(index % seg_size));
    } else {
        const std::size_t seg_size = longTranMesh.size();
        return from_longTran_vert(longTranMesh[index % seg_size], vertAxis->at(index / seg_size));
    }
}

std::size_t ExtrudedTriangularMesh3D::size() const {
    return longTranMesh.size() * vertAxis->size();
}

bool ExtrudedTriangularMesh3D::empty() const {
    return longTranMesh.empty() || vertAxis->empty();
}

void ExtrudedTriangularMesh3D::writeXML(XMLElement &object) const {
    object.attr("type", "extruded_triangular2d");
    { auto a = object.addTag("vert"); vertAxis->writeXML(a); }
    { auto a = object.addTag("long_tran"); longTranMesh.writeXML(a); }
}




}   // namespace plask
