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
/**
 * \file
 * Sample solver header for your solver
 */
#ifndef PLASK__SOLVER_YOUR_SOLVER_H
#define PLASK__SOLVER_YOUR_SOLVER_H

#include <plask/plask.hpp>

namespace plask { namespace ccategory { namespace your_solver {

/**
 * This is Doxygen documentation of your solver.
 * Write a brief description of it.
 */
struct YourSolver: public SolverWithMesh<ForExample_Geometry2DCartesian, ForExample_RectilinearMesh2D> {

    /// Sample receiver for temperature.
    ReceiverFor<Temperature, Geometry2DCartesian> inTemperature;

    /// Sample provider for simple value.
    ProviderFor<SomeSingleValueProperty>::WithValue outSingleValue;

    /// Sample provider for field (it's better to use delegate here).
    ProviderFor<SomeFieldProperty, Geometry2DCartesian>::Delegate outSomeField;

    YourSolver(const std::string& name="");

    virtual std::string getClassName() const { return "NameOfYourSolver"; }

    virtual std::string getClassDescription() const {
        return "This solver does this and that. And this description can be e.g. shown as a hint in GUI.";
    }

    virtual void loadConfiguration(plask::XMLReader& reader, plask::Manager& manager);

    /**
     * This method performs the main computations.
     * Add short description of every method in a comment like this.
     * \param parameter some value to be provided by the user for computationsś
     **/
    void compute(double parameter);

  protected:

    /// This is field data computed by this solver on its own mesh
    DataVector<double> my_data;

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    /// Method computing the value for the delegate provider
    const DataVector<const double> getDelegated(const MeshD<2>& dst_mesh, InterpolationMethod method=INTERPOLATION_DEFAULT);

};


}}} // namespace

#endif

