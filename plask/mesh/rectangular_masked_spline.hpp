/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__MESH_RECTANGULARMASKEDSPLINE_H
#define PLASK__MESH_RECTANGULARMASKEDSPLINE_H

#include "../math.hpp"
#include "rectangular_masked2d.hpp"
#include "rectangular_masked3d.hpp"
#include "interpolation.hpp"

namespace plask {

template <typename DstT, typename SrcT>
struct SplineMaskedRect2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh2D, const SrcT>
{
    typedef RectangularMaskedMesh2D MeshType;

    DataVector<SrcT> diff0, diff1;

    SplineMaskedRect2DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh2D>& src_mesh,
                                   const DataVector<const SrcT>& src_vec,
                                   const shared_ptr<const MeshD<2>>& dst_mesh,
                                   const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
        diff0(src_mesh->size()), diff1(src_mesh->size()) {}

    DstT at(std::size_t index) const override;
};

template <typename DstT, typename SrcT>
struct SplineMaskedRectElement2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh2D::ElementMesh, const SrcT>
{
    typedef RectangularMaskedMesh2D::ElementMesh MeshType;

    DataVector<SrcT> diff0, diff1;

    SplineMaskedRectElement2DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh2D::ElementMesh>& src_mesh,
                                          const DataVector<const SrcT>& src_vec,
                                          const shared_ptr<const MeshD<2>>& dst_mesh,
                                          const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh2D::ElementMesh, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
        diff0(src_mesh->size()), diff1(src_mesh->size()) {}

    DstT at(std::size_t index) const override;
};


template <typename DstT, typename SrcT>
struct SplineMaskedRect3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh3D, const SrcT>
{
    typedef RectangularMaskedMesh3D MeshType;

    DataVector<SrcT> diff0, diff1, diff2;

    SplineMaskedRect3DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh3D>& src_mesh,
                                   const DataVector<const SrcT>& src_vec,
                                   const shared_ptr<const MeshD<3>>& dst_mesh,
                                   const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh3D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
        diff0(src_mesh->size()), diff1(src_mesh->size()), diff2(src_mesh->size()) {}

    DstT at(std::size_t index) const override;
};

template <typename DstT, typename SrcT>
struct SplineMaskedRectElement3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh3D::ElementMesh, const SrcT>
{
    typedef RectangularMaskedMesh3D::ElementMesh MeshType;

    DataVector<SrcT> diff0, diff1, diff2;

    SplineMaskedRectElement3DLazyDataImpl(const shared_ptr<const RectangularMaskedMesh3D::ElementMesh>& src_mesh,
                                          const DataVector<const SrcT>& src_vec,
                                          const shared_ptr<const MeshD<3>>& dst_mesh,
                                          const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, RectangularMaskedMesh3D::ElementMesh, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
        diff0(src_mesh->size()), diff1(src_mesh->size()), diff2(src_mesh->size()) {}

    DstT at(std::size_t index) const override;
};


template <typename DstT, typename SrcT, typename BaseT=SplineMaskedRect2DLazyDataImpl<DstT, SrcT>>
struct PLASK_API HymanSplineMaskedRect2DLazyDataImpl: public BaseT
{
    HymanSplineMaskedRect2DLazyDataImpl(const shared_ptr<const typename BaseT::MeshType>& src_mesh,
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

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMaskedMesh2D::ElementMesh, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMaskedMesh2D::ElementMesh>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<2>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineMaskedRect2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type,
                                                       SplineMaskedRectElement2DLazyDataImpl<DstT, SrcT>>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


template <typename DstT, typename SrcT, typename BaseT=SplineMaskedRect3DLazyDataImpl<DstT, SrcT>>
struct PLASK_API HymanSplineMaskedRect3DLazyDataImpl: public BaseT {

    HymanSplineMaskedRect3DLazyDataImpl(const shared_ptr<const typename BaseT::MeshType>& src_mesh,
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

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<RectangularMaskedMesh3D::ElementMesh, SrcT, DstT, INTERPOLATION_SPLINE> {
    static LazyData<DstT> interpolate(const shared_ptr<const RectangularMaskedMesh3D::ElementMesh>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return new HymanSplineMaskedRect3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type,
                                                       SplineMaskedRectElement3DLazyDataImpl<DstT, SrcT>>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};

} // namespace plask

#endif // PLASK__MESH_RECTANGULARMASKEDSPLINE_H
