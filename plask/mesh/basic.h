#ifndef PLASK__MESH__BASIC_H
#define PLASK__MESH__BASIC_H

/** @file
This file includes some basic meshes.
@see @ref meshes
*/

#include "mesh.h"

namespace plask {

/// Mesh which represent set with only one point in space with size given as template parameter .
template <int DIM>
struct OnePointMesh: public plask::MeshD<DIM> {

    /// Held point:
    Vec<DIM, double> point;

    OnePointMesh(const plask::Vec<DIM, double>& point)
    : point(point) {}

    // plask::MeshD<DIM> methods implementation:

    virtual std::size_t size() const override {
        return 1;
    }

    virtual plask::Vec<DIM, double> at(std::size_t index) const override {
        return point;
    }

    virtual void writeXML(XMLElement& object) const override;

};

template <int DIM>
inline OnePointMesh<DIM> toMesh(const plask::Vec<DIM, double>& point) {
    return OnePointMesh<DIM>(point);
}

}   // namespace plask

#endif // PLASK__MESH__BASIC_H
