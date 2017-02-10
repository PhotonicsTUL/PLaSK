#include "equilateral3d.h"
#include "equilateral3d.h"

namespace plask {

EquilateralMesh3D::EquilateralMesh3D(IterationOrder iterationOrder, Vec<3> vec0, Vec<3> vec1, Vec<3> vec2):
    RectilinearMesh3D(iterationOrder), vec0(vec0), vec1(vec1), vec2(vec2) {}

EquilateralMesh3D::EquilateralMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder,
                                     Vec<3> vec0, Vec<3> vec1, Vec<3> vec2):
    RectilinearMesh3D(std::move(mesh0), std::move(mesh1), std::move(mesh2), iterationOrder), vec0(vec0), vec1(vec1), vec2(vec2) {}

EquilateralMesh3D::EquilateralMesh3D(const EquilateralMesh3D& src): RectilinearMesh3D(src), vec0(src.vec0), vec1(src.vec1), vec2(src.vec2) {}

void EquilateralMesh3D::writeXML(XMLElement& object) const {
    throw NotImplemented("writeXML for equilateral mesh");
}

shared_ptr<EquilateralMesh3D> EquilateralMesh3D::getMidpointsMesh() {
    return plask::make_shared<EquilateralMesh3D>(axis0->getMidpointsMesh(), axis1->getMidpointsMesh(), axis2->getMidpointsMesh(),
                                                 getIterationOrder(), vec0, vec0, vec0);
}

} // # namespace plask
