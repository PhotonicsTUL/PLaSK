#ifndef PLASK__MESH_RECTANGULARSPLINE_H
#define PLASK__MESH_RECTANGULARSPLINE_H

#include "../math.h"
#include "rectangular2d.h"
#include "rectangular3d.h"
#include "interpolation.h"

namespace plask {

template <typename DstT, typename SrcT>
struct SplineRect2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMesh<2>, const SrcT > {

    DataVector<SrcT> diff0, diff1;

    SplineRect2DLazyDataImpl(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<2>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<2>, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<2>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few SplineRect2DLazyDataImpl variants and choose one here
        return new SplineRect2DLazyDataImpl<typename std::remove_const<DstT>::type, typename std::remove_const<SrcT>::type>(src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT>
struct SplineRect3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMesh<3>, const SrcT > {

    DataVector<SrcT> diff0, diff1, diff2;

    SplineRect3DLazyDataImpl(const shared_ptr<const RectangularMesh<3>>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<3>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMesh<3>, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMesh<3>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        // You can have few SplineRect3DLazyDataImpl variants and choose one here
        return new SplineRect3DLazyDataImpl<typename std::remove_const<DstT>::type, typename std::remove_const<SrcT>::type>(src_mesh, src_vec, dst_mesh, flags);
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARSPLINE_H
