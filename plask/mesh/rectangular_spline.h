#ifndef PLASK__MESH_RECTANGULARSPLINE_H
#define PLASK__MESH_RECTANGULARSPLINE_H

#include "../math.h"
#include "rectangular2d.h"
#include "rectilinear3d.h"
#include "rectangular3d.h"
#include "equilateral3d.h"
#include "interpolation.h"

namespace plask {

template <typename DstT, typename SrcT>
struct SplineRect2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMesh<2>, const SrcT>
{
    DataVector<SrcT> diff0, diff1;

    SplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<2>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename DstT, typename SrcT>
struct SplineRect3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectilinearMesh3D, const SrcT>
{
    DataVector<SrcT> diff0, diff1, diff2;

    SplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<3>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineRect2DLazyDataImpl: public SplineRect2DLazyDataImpl<DstT, SrcT>
{
    HymanSplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                  const DataVector<const SrcT>& src_vec,
                                  const shared_ptr<const MeshD<2>>& dst_mesh,
                                  const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2>, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few HymanSplineRect2DLazyDataImpl variants and choose one here
        return new HymanSplineRect2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                 typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineRect3DLazyDataImpl: public SplineRect3DLazyDataImpl<DstT, SrcT> {

    HymanSplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                                  const DataVector<const SrcT>& src_vec,
                                  const shared_ptr<const MeshD<3>>& dst_mesh,
                                  const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few HymanSplineRect3DLazyDataImpl variants and choose one here
        return new HymanSplineRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                 typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<EquilateralMesh3D, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const EquilateralMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few HymanSplineRect3DLazyDataImpl variants and choose one here
        return new HymanSplineRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                 typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, EquilateralMesh3D::Transformed(src_mesh, dst_mesh), flags);
    }

};


template <typename DstT, typename SrcT>
struct PLASK_API SmoothSplineRect2DLazyDataImpl: public SplineRect2DLazyDataImpl<DstT, SrcT>
{
    SmoothSplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                   const DataVector<const SrcT>& src_vec,
                                   const shared_ptr<const MeshD<2>>& dst_mesh,
                                   const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2>, SrcT, DstT, INTERPOLATION_SMOOTH_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new SmoothSplineRect2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                  typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT>
struct PLASK_API SmoothSplineRect3DLazyDataImpl: public SplineRect3DLazyDataImpl<DstT, SrcT> {

    SmoothSplineRect3DLazyDataImpl(const shared_ptr<const RectilinearMesh3D>& src_mesh,
                                   const DataVector<const SrcT>& src_vec,
                                   const shared_ptr<const MeshD<3>>& dst_mesh,
                                   const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_SMOOTH_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few SmoothSplineRect3DLazyDataImpl variants and choose one here
        return new SmoothSplineRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                  typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<EquilateralMesh3D, SrcT, DstT, INTERPOLATION_SMOOTH_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const EquilateralMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new SmoothSplineRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                  typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, EquilateralMesh3D::Transformed(src_mesh, dst_mesh), flags);
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARSPLINE_H
