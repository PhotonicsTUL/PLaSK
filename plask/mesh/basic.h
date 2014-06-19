#ifndef PLASK__MESH__BASIC_H
#define PLASK__MESH__BASIC_H

/** @file
This file contains some basic meshes.
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
inline shared_ptr<OnePointMesh<DIM>> toMesh(const plask::Vec<DIM, double>& point) {
    return make_shared<OnePointMesh<DIM>>(point);
}

template<> void OnePointMesh<2>::writeXML(XMLElement& object) const;
template<> void OnePointMesh<3>::writeXML(XMLElement& object) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(OnePointMesh<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(OnePointMesh<3>)

/**
 * Mesh which trasnlate another mesh by given vector.
 */
template <int DIM>
struct TranslatedMesh: public MeshD<DIM> {

    Vec<DIM, double> translation;

    const shared_ptr<const MeshD<DIM>> sourceMesh;

    TranslatedMesh(const shared_ptr<const MeshD<DIM>>& sourceMesh, const Vec<DIM, double>& translation)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<DIM, double> at(std::size_t index) const override {
        return sourceMesh->at(index) + translation;
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslatedMesh<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslatedMesh<3>)

//TODO return special type for rectangular meshes
template <int DIM>
inline shared_ptr<TranslatedMesh<DIM>> translate(const shared_ptr<const MeshD<DIM>>& sourceMesh, const Vec<DIM, double>& translation) {
    return make_shared<TranslatedMesh<DIM>>(sourceMesh, translation);
}

}   // namespace plask

#endif // PLASK__MESH__BASIC_H
