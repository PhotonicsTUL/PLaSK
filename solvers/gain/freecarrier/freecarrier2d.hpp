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
#ifndef PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER2D_HPP
#define PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER2D_HPP

#include "freecarrier.hpp"

namespace plask { namespace gain { namespace freecarrier {

/**
 * Gain solver using Fermi Golden Rule
 */
template <typename GeometryT>
struct PLASK_SOLVER_API FreeCarrierGainSolver2D : public FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>> {
    using typename FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>>::ActiveRegionInfo;
    using typename FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>>::ActiveRegionParams;
    using typename FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>>::GeometryType;
    using typename FreeCarrierGainSolver<SolverWithMesh<GeometryT, MeshAxis>>::GainSpectrumType;

    FreeCarrierGainSolver2D(const std::string& name = "");

    std::string getClassName() const override;

  protected:
    void detectActiveRegions() override;

    template <typename DT> struct DataBase;
    struct ComputedData;
    struct GainData;
    struct DgdnData;
    struct EnergyLevelsData;

    typedef DataBase<Tensor2<double>> DataBaseTensor2;
    typedef DataBase<std::vector<double>> DataBaseVector;

    const LazyData<Tensor2<double>> getGainData(Gain::EnumType what,
                                                const shared_ptr<const MeshD<2>>& dst_mesh,
                                                double wavelength,
                                                InterpolationMethod interp = INTERPOLATION_DEFAULT) override;

    const LazyData<std::vector<double>> getEnergyLevels(EnergyLevels::EnumType which,
                                                        const shared_ptr<const MeshD<2>>& dst_mesh,
                                                        InterpolationMethod interp = INTERPOLATION_DEFAULT) override;
};

}}}  // namespace plask::gain::freecarrier

#endif  // PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER2D_HPP
