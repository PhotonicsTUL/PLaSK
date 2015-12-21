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
struct PLASK_API OnePointMesh: public plask::MeshD<DIM> {

	typedef Vec<DIM, double> DVec;

    /// Held point:
	DVec point;

    OnePointMesh(const plask::Vec<DIM, double>& point)
    : point(point) {}

    // plask::MeshD<DIM> methods implementation:

    std::size_t size() const override;

    DVec at(std::size_t index) const override;

    void writeXML(XMLElement& object) const override;

};

/**
 * Create one-point mesh that includes given @p point.
 * @param point point to include
 * @return mesh with one point
 */
template <int DIM>
inline shared_ptr<OnePointMesh<DIM>> toMesh(const plask::Vec<DIM, double>& point) {
    return plask::make_shared<OnePointMesh<DIM>>(point);
}

template<> void OnePointMesh<2>::writeXML(XMLElement& object) const;
template<> void OnePointMesh<3>::writeXML(XMLElement& object) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(OnePointMesh<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(OnePointMesh<3>)

/**
 * Mesh which trasnlate another mesh by given vector.
 */
template <int DIM>
struct PLASK_API TranslatedMesh: public MeshD<DIM> {

	typedef Vec<DIM, double> DVec;

	DVec translation;

    const shared_ptr<const MeshD<DIM>> sourceMesh;

    TranslatedMesh(const shared_ptr<const MeshD<DIM>>& sourceMesh, const Vec<DIM, double>& translation)
        : translation(translation), sourceMesh(sourceMesh) {}

	DVec at(std::size_t index) const override;

    std::size_t size() const override;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslatedMesh<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(TranslatedMesh<3>)

//TODO return special type for rectangular meshes
template <int DIM>
inline shared_ptr<TranslatedMesh<DIM>> translate(const shared_ptr<const MeshD<DIM>>& sourceMesh, const Vec<DIM, double>& translation) {
    return plask::make_shared<TranslatedMesh<DIM>>(sourceMesh, translation);
}

}   // namespace plask

#endif // PLASK__MESH__BASIC_H
