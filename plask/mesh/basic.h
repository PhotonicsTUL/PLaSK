#ifndef PLASK__MESH__BASIC_H
#define PLASK__MESH__BASIC_H

/** @file
This file includes some basic meshes.
@see @ref meshes
*/

#include "mesh.h"

namespace plask {

/// Mesh which represent set with only one point in space with size given as template parameter @p DIM.
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

extern template struct OnePointMesh<2>;
extern template struct OnePointMesh<3>;

/**
 * Mesh which trasnlate another mesh by given vector.
 */
template <int DIM>
struct TranslatedMesh: public MeshD<DIM> {

    Vec<DIM, double> translation;

    const MeshD<DIM>& sourceMesh;

    TranslatedMesh(const MeshD<DIM>& sourceMesh, const Vec<DIM, double>& translation)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<DIM, double> at(std::size_t index) const override {
        return sourceMesh.at(index) + translation;
    }

    virtual std::size_t size() const override {
        return sourceMesh.size();
    }

};

template <int DIM>
inline TranslatedMesh<DIM> translate(const MeshD<DIM>& sourceMesh, const Vec<DIM, double>& translation) {
    return TranslatedMesh<DIM>(sourceMesh, translation);
}

}   // namespace plask

#endif // PLASK__MESH__BASIC_H
