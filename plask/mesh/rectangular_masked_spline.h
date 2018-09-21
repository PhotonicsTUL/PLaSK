#ifndef PLASK__MESH_RECTANGULARMASKEDSPLINE_H
#define PLASK__MESH_RECTANGULARMASKEDSPLINE_H

#include "../math.h"
#include "rectangular_masked2d.h"
#include "rectangular_masked3d.h"
#include "interpolation.h"

namespace plask {

template <typename DstT, typename SrcT>
struct SplineMaskedRect2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh2D, const SrcT>
{
    DataVector<SrcT> diff0, diff1;

    SplineMaskedRect2DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh2D>& src_mesh,
                             const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<2>>& dst_mesh,
                             const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename DstT, typename SrcT>
struct SplineMaskedRect3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh3D, const SrcT>
{
    DataVector<SrcT> diff0, diff1, diff2;

    SplineMaskedRect3DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh3D>& src_mesh,
                                     const DataVector<const SrcT>& src_vec,
                                     const shared_ptr<const MeshD<3>>& dst_mesh,
                                     const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl: public SplineMaskedRect2DLazyDataImpl<DstT, SrcT>
{
    HymanSplineMaskedRect2DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh2D>& src_mesh,
                                          const DataVector<const SrcT>& src_vec,
                                          const shared_ptr<const MeshD<2>>& dst_mesh,
                                          const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMaskedMesh2D, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMaskedMesh2D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineMaskedRect2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                         typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT>
struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl: public SplineMaskedRect3DLazyDataImpl<DstT, SrcT> {

    HymanSplineMaskedRect3DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh3D>& src_mesh,
                                  const DataVector<const SrcT>& src_vec,
                                  const shared_ptr<const MeshD<3>>& dst_mesh,
                                  const InterpolationFlags& flags);
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMaskedMesh3D, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMaskedMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineMaskedRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                         typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


} // namespace plask

#endif // PLASK__MESH_RECTANGULARMASKEDSPLINE_H
