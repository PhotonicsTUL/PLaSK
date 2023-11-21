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
#ifndef PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER3D_HPP
#define PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER3D_HPP

#include "freecarrier.hpp"

namespace plask { namespace gain { namespace freecarrier {

/**
 * Gain solver using Fermi Golden Rule
 */
struct PLASK_SOLVER_API FreeCarrierGainSolver3D : public FreeCarrierGainSolver<SolverOver<Geometry3D>> {
    using typename FreeCarrierGainSolver<SolverOver<Geometry3D>>::ActiveRegionInfo;
    using typename FreeCarrierGainSolver<SolverOver<Geometry3D>>::ActiveRegionParams;
    using typename FreeCarrierGainSolver<SolverOver<Geometry3D>>::GeometryType;
    using typename FreeCarrierGainSolver<SolverOver<Geometry3D>>::GainSpectrumType;

    FreeCarrierGainSolver3D(const std::string& name = "");

    std::string getClassName() const override;

  private:
    struct Region {
        size_t bottom, top, lon, tra;
        std::vector<bool> isQW;
        std::vector<shared_ptr<Material>> materials;
        Region() {}
        Region(size_t b,
               size_t t,
               size_t x,
               size_t y,
               const std::vector<bool>& isQW,
               const std::vector<shared_ptr<Material>>& materials)
            : bottom(b),
              top(t),
              lon(x),
              tra(y),
              isQW(isQW),
              materials(materials.begin() + bottom - 1, materials.begin() + top + 1) {}
    };

  protected:
    void detectActiveRegions() override;

    template <typename DT> struct DataBase;
    struct ComputedData;
    struct GainData;
    struct DgdnData;
    struct EnergyLevelsData;

    typedef DataBase<Tensor2<double>> DataBaseTensor2;
    typedef DataBase<std::vector<double>> DataBaseVector;

    friend struct ActiveRegionMesh;

    const LazyData<Tensor2<double>> getGainData(Gain::EnumType what,
                                                const shared_ptr<const MeshD<3>>& dst_mesh,
                                                double wavelength,
                                                InterpolationMethod interp = INTERPOLATION_DEFAULT) override;

    const LazyData<std::vector<double>> getEnergyLevels(EnergyLevels::EnumType which,
                                                        const shared_ptr<const MeshD<3>>& dst_mesh,
                                                        InterpolationMethod interp = INTERPOLATION_DEFAULT) override;
};

}}}  // namespace plask::gain::freecarrier

#endif  // PLASK__SOLVER__GAIN_FREECARRIER_FREECARRIER3D_HPP
