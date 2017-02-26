#ifndef PLASK__MESH_EQUILATERAL3D_H
#define PLASK__MESH_EQUILATERAL3D_H

/** @file
This file contains equilateral mesh for 3D space.
*/

#include "rectilinear3d.h"

namespace plask {

class PLASK_API EquilateralMesh3D: public RectilinearMesh3D {

  protected:    
    double trans[9];    ///< Transfromation matrix

  private:
    double inv[9];      ///< Inverse of the transformation matrix

    /// Find inverse of the transformation matrix
    void findInverse() {
        double idet = 1. / (trans[0] * (trans[4]*trans[8] - trans[5]*trans[7]) +
                            trans[1] * (trans[5]*trans[6] - trans[3]*trans[8]) +
                            trans[2] * (trans[3]*trans[7] - trans[4]*trans[6]));
        inv[0] = idet * ( trans[4]*trans[8] - trans[5]*trans[7]);
        inv[1] = idet * (-trans[1]*trans[8] + trans[2]*trans[7]);
        inv[2] = idet * ( trans[1]*trans[5] - trans[2]*trans[4]);
        inv[3] = idet * (-trans[3]*trans[8] + trans[5]*trans[6]);
        inv[4] = idet * ( trans[0]*trans[8] - trans[2]*trans[6]);
        inv[5] = idet * (-trans[0]*trans[5] + trans[2]*trans[3]);
        inv[6] = idet * ( trans[3]*trans[7] - trans[4]*trans[6]);
        inv[7] = idet * (-trans[0]*trans[7] + trans[1]*trans[6]);
        inv[8] = idet * ( trans[0]*trans[4] - trans[1]*trans[3]);
    }
    
  public:
      
    /// Adapter for the destination mesh in interpolations, moving the src point to mesh coordinates
    struct Transformed: public MeshD<3> {
        const EquilateralMesh3D* src;
        shared_ptr<const MeshD<3>> dst;
        Transformed(const EquilateralMesh3D* src, const shared_ptr<const MeshD<3>>& dst):
            src(src), dst(dst) {}
        Transformed(const shared_ptr<const EquilateralMesh3D>& src, const shared_ptr<const MeshD<3>>& dst):
            src(src.get()), dst(dst) {}
        size_t size() const override { return dst->size(); }
        Vec<3,double> at(size_t index) const override {
            return src->toMeshCoords(dst->at(index));
        }
    };
      
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

    /// Return the first mesh vector
    Vec<3,double> getVec0() const {
        return vec<double>(trans[0], trans[3], trans[6]);
    }
    
    /// Return the second mesh vector
    Vec<3,double> getVec1() const {
        return vec<double>(trans[1], trans[4], trans[7]);
    }

    /// Return the third mesh vector
    Vec<3,double> getVec2() const {
        return vec<double>(trans[2], trans[5], trans[8]);
    }
    
    /// Return the first inverse vector
    Vec<3,double> getInvVec0() const {
        return vec<double>(inv[0], inv[1], inv[2]);
    }
    
    /// Return the second inverse vector
    Vec<3,double> getInvVec1() const {
        return vec<double>(inv[3], inv[4], inv[5]);
    }
    
    /// Return the third inverse vector
    Vec<3,double> getInvVec2() const {
        return vec<double>(inv[6], inv[7], inv[8]);
    }
    
    /// Set the first mesh vector
    void setVec0(Vec<3,double> vec0) {
        trans[0] = vec0.c0;
        trans[3] = vec0.c1;
        trans[6] = vec0.c2;
        findInverse();
        fireChanged(Event::EVENT_USER_DEFINED);
    }
    
    /// Set the second mesh vector
    void setVec1(Vec<3,double> vec1) {
        trans[1] = vec1.c0;
        trans[4] = vec1.c1;
        trans[7] = vec1.c2;
        findInverse();
        fireChanged(Event::EVENT_USER_DEFINED);
    }
    
    /// Set the third mesh vector
    void setVec2(Vec<3,double> vec2) {
        trans[2] = vec2.c0;
        trans[5] = vec2.c1;
        trans[8] = vec2.c2;
        findInverse();
        fireChanged();
    }

    /**
     * Transform point in the real coordinates to mesh coordinates
     * \param point point to transfrom
     * \returns transfromed coordinates
     */
    Vec<3,double> toMeshCoords(Vec<3,double> point) const {
        return Vec<3,double>(inv[0] * point.c0 + inv[1] * point.c1 + inv[2] * point.c2,
                             inv[3] * point.c0 + inv[4] * point.c1 + inv[5] * point.c2,
                             inv[6] * point.c0 + inv[7] * point.c1 + inv[8] * point.c2);
    }
    
    /**
     * Transform point in the real coordinates to mesh coordinates
     * \param c0,c1,c2 point to transfrom
     * \returns transfromed coordinates
     */
    Vec<3,double> toMeshCoords(double c0, double c1, double c2) const {
        return Vec<3,double>(inv[0] * c0 + inv[1] * c1 + inv[2] * c2,
                             inv[3] * c0 + inv[4] * c1 + inv[5] * c2,
                             inv[6] * c0 + inv[7] * c1 + inv[8] * c2);
    }

    /**
     * Transform point in mesh coordinates to the real coordinates
     * \param coords coordinates to transfrom
     * \returns transfromed coordinates
     */
    inline Vec<3,double> fromMeshCoords(Vec<3,double> coords) const {
        return Vec<3,double>(trans[0] * coords.c0 + trans[1] * coords.c1 + trans[2] * coords.c2,
                             trans[3] * coords.c0 + trans[4] * coords.c1 + trans[5] * coords.c2,
                             trans[6] * coords.c0 + trans[7] * coords.c1 + trans[8] * coords.c2);
    }
    
    /**
     * Transform point in mesh coordinates to the real coordinates
     * \param c0,c1,c2 coordinates to transfrom
     * \returns transfromed coordinates
     */
    inline Vec<3,double> fromMeshCoords(double c0, double c1, double c2) const {
        return Vec<3,double>(trans[0] * c0 + trans[1] * c1 + trans[2] * c2,
                             trans[3] * c0 + trans[4] * c1 + trans[5] * c2,
                             trans[6] * c0 + trans[7] * c1 + trans[8] * c2);
    }
    
    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @param index2 index of point in axis2
     * @return point with given @p index
     */
    Vec<3, double> at(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        return fromMeshCoords(axis0->at(index0), axis1->at(index1), axis2->at(index2));
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
    Vec<3,double> getElementMidpoint(std::size_t index0, std::size_t index1, std::size_t index2) const override {
        return fromMeshCoords(getElementMidpoint0(index0), getElementMidpoint1(index1), getElementMidpoint2(index2));
    }
};


template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<EquilateralMesh3D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const EquilateralMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis0->size() == 0 || src_mesh->axis1->size() == 0 || src_mesh->axis2->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new LinearInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>
            (src_mesh, src_vec, EquilateralMesh3D::Transformed(src_mesh, dst_mesh), flags);
    }
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<EquilateralMesh3D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const EquilateralMesh3D>& src_mesh, const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh, const InterpolationFlags& flags) {
        if (src_mesh->axis0->size() == 0 || src_mesh->axis1->size() == 0 || src_mesh->axis2->size() == 0)
            throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborInterpolatedLazyDataImpl<DstT, RectilinearMesh3D, SrcT>
            (src_mesh, src_vec, EquilateralMesh3D::Transformed(src_mesh, dst_mesh), flags);
    }
};



}; // namespace plask

#endif // PLASK__SMESH_EQUILATERAL3D_H
