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
#ifndef PLASK__MODULE_THERMAL_THERM2D_H
#define PLASK__MODULE_THERMAL_THERM2D_H

#include <plask/plask.hpp>
#include <plask/common/fem.hpp>

#include "common.hpp"

namespace plask { namespace thermal { namespace tstatic {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename Geometry2DType>
struct PLASK_SOLVER_API ThermalFem2DSolver : public FemSolverWithMaskedMesh<Geometry2DType, RectangularMesh<2>> {
  protected:
    int loopno;     ///< Number of completed loops
    double maxT;    ///< Maximum temperature recorded
    double toterr;  ///< Maximum estimated error during all iterations (useful for single calculations managed by external python
                    ///< script)

    DataVector<double> temperatures;  ///< Computed temperatures

    DataVector<double> thickness;  ///< Thicknesses of the layers

    DataVector<Vec<2, double>> fluxes;  ///< Computed (only when needed) heat fluxes on our own mesh

    /// Set stiffness matrix + load vector
    void setMatrix(FemMatrix<>& A,
                   DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, Radiation>& bradiation);

    /// Setup matrix
    template <typename MatrixT> MatrixT makeMatrix();

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes();  // [W/m^2]

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

  public:
    // Boundary conditions
    BoundaryConditions<RectangularMesh<2>::Boundary, double>
        temperature_boundary;  ///< Boundary condition of constant temperature (K)
    BoundaryConditions<RectangularMesh<2>::Boundary, double>
        heatflux_boundary;  ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectangularMesh<2>::Boundary, Convection> convection_boundary;  ///< Boundary condition of convection
    BoundaryConditions<RectangularMesh<2>::Boundary, Radiation> radiation_boundary;    ///< Boundary condition of radiation

    typename ProviderFor<Temperature, Geometry2DType>::Delegate outTemperature;

    typename ProviderFor<HeatFlux, Geometry2DType>::Delegate outHeatFlux;

    typename ProviderFor<ThermalConductivity, Geometry2DType>::Delegate outThermalConductivity;

    ReceiverFor<Heat, Geometry2DType> inHeat;

    double maxerr;    ///< Maximum residual error accepted as convergence
    double inittemp;  ///< Initial temperature

    /**
     * Run temperature calculations
     * \param loops maximum number of loops to run
     * \return max correction of temperature against the last call
     **/
    double compute(int loops = 1);

    /// Get max absolute correction for temperature
    double getErr() const { return toterr; }

    void loadConfiguration(XMLReader& source,
                           Manager& manager) override;  // for solver configuration (see: *.xpl file with structures)

    ThermalFem2DSolver(const std::string& name = "");

    std::string getClassName() const override;

    ~ThermalFem2DSolver();

  public:
    struct ThermalConductivityData : public LazyDataImpl<Tensor2<double>> {
        const ThermalFem2DSolver* solver;
        shared_ptr<const MeshD<2>> dest_mesh;
        InterpolationFlags flags;
        LazyData<double> temps;
        ThermalConductivityData(const ThermalFem2DSolver* solver, const shared_ptr<const MeshD<2>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const override;
        std::size_t size() const override;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) const;

    const LazyData<Vec<2>> getHeatFluxes(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method);
};

}}}  // namespace plask::thermal::tstatic

#endif
