#ifndef PLASK__MESH_EXTRUDED_TRIANGULAR3D_H
#define PLASK__MESH_EXTRUDED_TRIANGULAR3D_H

#include "axis1d.h"
#include "triangular2d.h"

namespace plask {

struct ExtrudedTriangularMesh3D: public MeshD<3> {

    const shared_ptr<MeshAxis> vertAxis;

    TriangularMesh2D longTranMesh;

    /// Iteration order, if true vert axis are changed the fastest, else it is changed the slowest.
    bool vertFastest;

    Vec<3, double> at(std::size_t index) const override;

    std::size_t size() const override;

    bool empty() const override;

    void writeXML(XMLElement& object) const override;
};

}   // namespace plask

#endif // PLASK__MESH_EXTRUDED_TRIANGULAR3D_H
