#ifndef PLASK__MESH_EQUILATERAL3D_H
#define PLASK__MESH_EQUILATERAL3D_H

/** @file
This file contains equilateral mesh for 3D space.
*/

#include "rectilinear3d.h"

namespace plask {

struct EquilateralMesh3D: public RectilinearMesh3D {

    /// First axis vector
    Vec<3> vec0;

    /// Second axis vector
    Vec<3> vec1;

    /// Third axis vector
    Vec<3> vec2;

    /**
     * Construct mesh which has all axes of type OrderedAxis and all are empty.
     * @param iterationOrder iteration order
     * @param vec0 first axis vector
     * @param vec1 second axis vector
     * @param vec2 third axis vector
     */
    explicit EquilateralMesh3D(IterationOrder iterationOrder = ORDER_012,
                               Vec<3> vec0 = vec(1., 0., 0.), Vec<3> vec1 = vec(0., 1., 0.), Vec<3> vec2 = vec(0., 0., 1.));

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     * @param iterationOrder iteration order
     * @param vec0 first axis vector
     * @param vec1 second axis vector
     * @param vec2 third axis vector
     */
    EquilateralMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder = ORDER_012,
                      Vec<3> vec0 = vec(1., 0., 0.), Vec<3> vec1 = vec(0., 1., 0.), Vec<3> vec2 = vec(0., 0., 1.));

    /// Copy constructor
    EquilateralMesh3D(const EquilateralMesh3D& src);

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        double c0 = axis0->at(index0), c1 = axis1->at(index1), c2 = axis2->at(index2);
        return Vec<3, double>(vec0.c0 * c0 + vec1.c0 * c1 + vec2.c0 * c2,
                              vec0.c1 * c0 + vec1.c1 * c1 + vec2.c1 * c2,
                              vec0.c2 * c0 + vec1.c2 * c1 + vec2.c2 * c2);
    }

    /**
     * Return a mesh that enables iterating over middle points of the elements
     * \return new equilateral mesh with points in the middles of original elements
     */
    shared_ptr<EquilateralMesh3D> getMidpointsMesh();

    /**
     * Get point in center of Elements.
     * @param index0, index1, index2 index of Elements
     * @return point in center of element with given index
     */
    Vec<3, double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        double c0 = getElementMidpoint0(index0), c1 = getElementMidpoint1(index1), c2 = getElementMidpoint2(index2);
        return Vec<3, double>(vec0.c0 * c0 + vec1.c0 * c1 + vec2.c0 * c2,
                              vec0.c1 * c0 + vec1.c1 * c1 + vec2.c1 * c2,
                              vec0.c2 * c0 + vec1.c2 * c1 + vec2.c2 * c2);
    }

};


}; // namespace plask

#endif // PLASK__SMESH_EQUILATERAL3D_H
