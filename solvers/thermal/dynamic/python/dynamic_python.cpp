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
#include <cmath>
#include <plask/python.hpp>
#include <plask/common/fem/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT2d.hpp"
#include "../femT3d.hpp"
using namespace plask::thermal::dynamic;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(dynamic)
{
    {CLASS(DynamicThermalFem2DSolver<Geometry2DCartesian>, "Dynamic2D",
        u8"Finite element thermal solver for 2D Cartesian geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, u8"Initial temperature (K)");
        RW_FIELD(timestep, u8"Time step (ns)");
        RW_FIELD(methodparam, u8"Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, u8"Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, u8"Frequency of rebuild mass");
        RW_FIELD(logfreq, u8"Frequency of iteration progress reporting");
        RO_PROPERTY(time, getElapsTime, u8"Time of calculations performed so far since the last solver invalidation.");
        RO_PROPERTY(elapsed_time, getElapsTime, u8"Alias for :attr:`time` (obsolete).");
        registerFemSolverWithMaskedMesh(solver);
    }

    {CLASS(DynamicThermalFem2DSolver<Geometry2DCylindrical>, "DynamicCyl",
        u8"Finite element thermal solver for 2D cylindrical geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, u8"Initial temperature (K)");
        RW_FIELD(timestep, u8"Time step (ns)");
        RW_FIELD(methodparam, u8"Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, u8"Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, u8"Frequency of rebuild mass");
        RW_FIELD(logfreq, u8"Frequency of iteration progress reporting");
        RO_PROPERTY(time, getElapsTime, u8"Time of calculations performed so far since the last solver invalidation.");
        RO_PROPERTY(elapsed_time, getElapsTime, u8"Alias for :attr:`time` (obsolete).");
        registerFemSolverWithMaskedMesh(solver);
    }

    {CLASS(DynamicThermalFem3DSolver, "Dynamic3D",
        u8"Finite element thermal solver for 3D Cartesian geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("time"));
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        RW_FIELD(inittemp, u8"Initial temperature (K)");
        RW_FIELD(timestep, u8"Time step (ns)");
        RW_FIELD(methodparam, u8"Initial parameter determining the calculation method: 0.5 - Crank-Nicolson method, 0 - explicit method, 1 - implicit method");
        RW_FIELD(lumping, u8"Chosen mass matrix type from lumped or non-lumped (consistent)");
        RW_FIELD(rebuildfreq, u8"Frequency of rebuild mass");
        solver.def_readwrite("algorithm", &__Class__::algorithm, u8"Chosen matrix factorization algorithm");
        solver.def_readwrite("logfreq", &__Class__::logfreq, u8"Frequency of iteration progress reporting");
        RO_PROPERTY(time, getElapsTime, u8"Time of calculations performed so far since the last solver invalidation.");
        RO_PROPERTY(elapsed_time, getElapsTime, u8"Alias for :attr:`time` (obsolete).");
        registerFemSolverWithMaskedMesh(solver);
    }

}
