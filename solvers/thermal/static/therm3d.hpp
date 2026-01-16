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
#ifndef PLASK__MODULE_THERMAL_TFem_H
#define PLASK__MODULE_THERMAL_TFem_H

#include <plask/plask.hpp>
#include <plask/common/fem.hpp>

#include "common.hpp"

namespace plask { namespace thermal { namespace tstatic {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct PLASK_SOLVER_API ThermalFem3DSolver: public FemSolverWithMaskedMesh<Geometry3D, RectangularMesh<3>> {

  protected:

    int loopno;         ///< Number of completed loops
    double maxT;        ///< Maximum temperature recorded
    double toterr;      ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)

    DataVector<double> temperatures;            ///< Computed temperatures

    DataVector<double> thickness;               ///< Thicknesses of the layers

    DataVector<Vec<3,double>> fluxes;           ///< Computed (only when needed) heat fluxes on our own mesh

    /**
     * Set stiffness matrix and load vector
     * \param[out] A matrix to fill-in
     * \param[out] B load vector
     * \param btemperature boundary conditions: constant temperature
     * \param bheatflux boundary conditions: constant heat flux
     * \param bconvection boundary conditions: convention
     * \param bradiation boundary conditions: radiation
     **/
    void setMatrix(FemMatrix<>& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,Radiation>& bradiation
                  );

    /**
     * Apply boundary conditions of the first kind
     */
    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary,double>& btemperature);

    /// Setup matrix
    template <typename MatrixT>
    MatrixT makeMatrix();

    /// Create 3D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

  public:

    double inittemp;    ///< Initial temperature
    double maxerr;     ///< Maximum temperature correction accepted as convergence

    // Boundary conditions
    BoundaryConditions<RectangularMesh<3>::Boundary,double> temperature_boundary;      ///< Boundary condition of constant temperature (K)
    BoundaryConditions<RectangularMesh<3>::Boundary,double> heatflux_boundary;         ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectangularMesh<3>::Boundary,Convection> convection_boundary;   ///< Boundary condition of convection
    BoundaryConditions<RectangularMesh<3>::Boundary,Radiation> radiation_boundary;     ///< Boundary condition of radiation

    typename ProviderFor<Temperature,Geometry3D>::Delegate outTemperature;

    typename ProviderFor<HeatFlux,Geometry3D>::Delegate outHeatFlux;

    typename ProviderFor<ThermalConductivity,Geometry3D>::Delegate outThermalConductivity;

    ReceiverFor<Heat,Geometry3D> inHeat;

    /**
     * Run temperature calculations
     * \param loops maximum number of loops to run
     * \return max correction of temperature against the last call
     **/
    double compute(int loops=1);

    void loadConfiguration(XMLReader& source, Manager& manager) override; // for solver configuration (see: *.xpl file with structures)

    ThermalFem3DSolver(const std::string& name="");

    std::string getClassName() const override { return "thermal.Static3D"; }

    ~ThermalFem3DSolver();

    /// Get max absolute correction for temperature
    double getErr() const { return toterr; }

    /// \return current algorithm
    FemMatrixAlgorithm getAlgorithm() const { return algorithm; }

    /**
     * Set algorithm
     * \param alg new algorithm
     */
    void setAlgorithm(FemMatrixAlgorithm alg);

  protected:

    struct ThermalConductivityData: public LazyDataImpl<Tensor2<double>> {
        const ThermalFem3DSolver* solver;
        shared_ptr<const MeshD<3>> dest_mesh;
        InterpolationFlags flags;
        LazyData<double> temps;
        ThermalConductivityData(const ThermalFem3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const override;
        std::size_t size() const override;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const;

    const LazyData<Vec<3>> getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method);
};

}}} //namespaces

#endif
