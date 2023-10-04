/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2023 Lodz University of Technology
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
#ifndef PLASK_MESH_LATERAL_HPP
#define PLASK_MESH_LATERAL_HPP

#include "axis1d.hpp"
#include "interpolation.hpp"
#include "mesh.hpp"

namespace plask {

namespace detail {
struct FlatMesh : MeshD<2> {
    shared_ptr<const MeshD<3>> originalMesh;

    FlatMesh(const shared_ptr<const MeshD<3>>& originalMesh) : originalMesh(originalMesh) {}

    std::size_t size() const override { return originalMesh->size(); }

    plask::Vec<2> at(std::size_t index) const override {
        auto point = originalMesh->at(index);
        return Vec<2>(point.c0, point.c1);
    }
};
}  // namespace detail

/**
 * 3D mesh with arbitrary lateral mesh and constant vertical coordinate.
 */
template <typename MeshT> struct LateralMesh3D : MeshD<3> {
    shared_ptr<MeshT> lateralMesh;
    double vert;

    LateralMesh3D(const shared_ptr<MeshT>& lateralMesh, double vert = 0.) : lateralMesh(lateralMesh), vert(vert) {}

    std::size_t size() const override { return lateralMesh->size(); }

    plask::Vec<3> at(std::size_t index) const override {
        Vec<2> p = lateralMesh->at(index);
        return Vec<3>(p.c0, p.c1, vert);
    }

    template <typename T = MeshT> shared_ptr<LateralMesh3D<typename T::ElementMesh>> getElementMesh() const {
        return make_shared<LateralMesh3D<typename T::ElementMesh>>(lateralMesh->getElementMesh(), vert);
    }
};

template <typename SrcMeshT, typename SrcT, typename DstT, InterpolationMethod method>
struct InterpolationAlgorithm<LateralMesh3D<SrcMeshT>, SrcT, DstT, method> {
    static LazyData<DstT> interpolate(const shared_ptr<const LateralMesh3D<SrcMeshT>>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags) {
        return InterpolationAlgorithm<SrcMeshT, SrcT, DstT, method>::interpolate(
            src_mesh->lateralMesh, src_vec, make_shared<const detail::FlatMesh>(dst_mesh), flags);
    }
};

template <typename SrcMeshT, typename SrcT, typename DstT>
struct InterpolationAlgorithm<LateralMesh3D<SrcMeshT>, SrcT, DstT, INTERPOLATION_DEFAULT> {
    static LazyData<DstT> interpolate(const shared_ptr<const LateralMesh3D<SrcMeshT>>&,
                                      const DataVector<const SrcT>&,
                                      const shared_ptr<const MeshD<3>>&,
                                      const InterpolationFlags& PLASK_UNUSED(flags)) {
        throw CriticalException(
            "interpolate(...) called for INTERPOLATION_DEFAULT method. Contact solver author to fix this issue."
#ifndef NDEBUG
            "\n\nINFO FOR SOLVER AUTHOR: To avoid this error use "
            "'getInterpolationMethod<YOUR_DEFAULT_METHOD>(interpolation_method) in C++ code of the provider in your solver.\n"
#endif
        );
    }
};

/**
 * 3D mesh with arbitrary lateral mesh and vertical axis.
 */
template <typename MeshT> struct MultiLateralMesh3D : MeshD<3> {
    shared_ptr<MeshT> lateralMesh;
    shared_ptr<MeshAxis> vertAxis;

    MultiLateralMesh3D(const shared_ptr<MeshT>& lateralMesh, const shared_ptr<MeshAxis>& vert)
        : lateralMesh(lateralMesh), vertAxis(vert) {}

    std::size_t size() const override { return lateralMesh->size() * vertAxis->size(); }

    plask::Vec<3> at(std::size_t index) const override {
        size_t i = index / vertAxis->size(), j = index % vertAxis->size();
        Vec<2> p = lateralMesh->at(i);
        return Vec<3>(p.c0, p.c1, vertAxis->at(j));
    }

    template <typename T = MeshT> shared_ptr<MultiLateralMesh3D<typename T::ElementMesh>> getElementMesh() const {
        return shared_ptr<MultiLateralMesh3D<typename T::ElementMesh>>(
            new MultiLateralMesh3D<typename T::ElementMesh>(lateralMesh->getElementMesh(), vertAxis->getMidpointAxis()));
    }
};

}  // namespace plask

#endif
