#ifndef PLASK__MESH_RECTANGULARFILTEREDSPLINE_H
#define PLASK__MESH_RECTANGULARFILTEREDSPLINE_H

#include "../math.h"
#include "rectangular_filtered2d.h"
#include "rectangular_filtered3d.h"
#include "interpolation.h"

namespace plask {

template <typename DstT, typename SrcT>
struct SplineFilteredRect2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularFilteredMesh2D, const SrcT>
{
    DataVector<SrcT> diff0, diff1;

    SplineFilteredRect2DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<2>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename DstT, typename SrcT>
struct SplineFilteredRect3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularFilteredMesh3D, const SrcT>
{
    DataVector<SrcT> diff0, diff1, diff2;

    SplineFilteredRect3DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh,
                                     const DataVector<const SrcT>& src_vec,
                                     const shared_ptr<const MeshD<3>>& dst_mesh,
                                     const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineFilteredRect2DLazyDataImpl: public SplineFilteredRect2DLazyDataImpl<DstT, SrcT>
{
    HymanSplineFilteredRect2DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh,
                                          const DataVector<const SrcT>& src_vec,
                                          const shared_ptr<const MeshD<2>>& dst_mesh,
                                          const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh2D, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh2D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineFilteredRect2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                         typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineFilteredRect3DLazyDataImpl: public SplineFilteredRect3DLazyDataImpl<DstT, SrcT> {

    HymanSplineFilteredRect3DLazyDataImpl(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh,
                                  const DataVector<const SrcT>& src_vec,
                                  const shared_ptr<const MeshD<3>>& dst_mesh,
                                  const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularFilteredMesh3D, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularFilteredMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineFilteredRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                         typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARFILTEREDSPLINE_H
