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

#include "../ddm2d.hpp"
using namespace plask::electrical::drift_diffusion;

template <typename GeometryT>
inline static void register_drift_diffusion_solver(const char* name, const char* geoname)
{
    typedef DriftDiffusionModel2DSolver<GeometryT>  __Class__;
    ExportSolver<DriftDiffusionModel2DSolver<GeometryT>> solver(name, format(

        u8"{0}(name=\"\")\n\n"

        u8"Finite element drift-diffusion electrical solver for 2D {1} geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    METHOD(compute, compute, u8"Run drift-diffusion calculations", py::arg("loops")=0);
    METHOD(get_total_current, getTotalCurrent, u8"Get total current flowing through active region [mA]", py::arg("nact")=0);
    METHOD(find_energy_levels, findEnergyLevels, u8"Run energy levels calculations - TEST");
    //METHOD(integrate_current, integrateCurrent, u8"Integrate vertical total current at certain level [mA]", py::arg("vindex"), py::arg("onlyactive")=false);
    /*RO_PROPERTY(err, getErr, u8"Maximum estimated error");*/
    RECEIVER(inTemperature, u8"");
    PROVIDER(outPotential, u8"");
    PROVIDER(outFermiLevels, u8"");
    PROVIDER(outBandEdges, u8"");
    PROVIDER(outCurrentDensityForElectrons, u8"");
    PROVIDER(outCurrentDensityForHoles, u8"");
    PROVIDER(outCarriersConcentration, u8"");
    PROVIDER(outHeat, u8"");
    /*PROVIDER(outConductivity, u8"");*/
    BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
    solver.def_readwrite("maxerrVi", &__Class__::maxerrPsiI, u8"Limit for the initial potential estimate updates");
    solver.def_readwrite("maxerrV0", &__Class__::maxerrPsi0, u8"Limit for the built-in potential updates");
    solver.def_readwrite("maxerrV", &__Class__::maxerrPsi, u8"Limit for the potential updates");
    solver.def_readwrite("maxerrFn", &__Class__::maxerrFn, u8"Limit for the electrons quasi-Fermi level updates");
    solver.def_readwrite("maxerrFp", &__Class__::maxerrFp, u8"Limit for the holes quasi-Fermi level updates");
    solver.def_readwrite("loopsVi", &__Class__::loopsPsiI, u8"Loops limit for the initial potential estimate");
    solver.def_readwrite("loopsV0", &__Class__::loopsPsi0, u8"Loops limit for the built-in potential");
    solver.def_readwrite("loopsV", &__Class__::loopsPsi, u8"Loops limit for the potential");
    solver.def_readwrite("loopsFn", &__Class__::loopsFn, u8"Loops limit for the electrons quasi-Fermi level");
    solver.def_readwrite("loopsFp", &__Class__::loopsFp, u8"Loops limit for the holes quasi-Fermi level");
    solver.def_readwrite("Rsrh", &__Class__::mRsrh, u8"True if SRH recombination is taken into account");
    solver.def_readwrite("Rrad", &__Class__::mRrad, u8"True if radiative recombination is taken into account");
    solver.def_readwrite("Raug", &__Class__::mRaug, u8"True if Auger recombination is taken into account");
    solver.def_readwrite("Pol", &__Class__::mPol, u8"True if polarization effects are taken into account");
    solver.def_readwrite("FullIon", &__Class__::mFullIon, u8"True if dopants are completely ionized");
    solver.def_readwrite("SchottkyP", &__Class__::mSchottkyP, u8"Schottky barrier for p-type constact");
    solver.def_readwrite("SchottkyN", &__Class__::mSchottkyN, u8"Schottky barrier for n-type constact");
    /*METHOD(get_electrostatic_energy, getTotalEnergy,
           u8"Get the energy stored in the electrostatic field in the analyzed structure.\n\n"
           u8"Return:\n"
           u8"    Total electrostatic energy [J].\n"
    );
    METHOD(get_capacitance, getCapacitance,
           u8"Get the structure capacitance.\n\n"
           u8"Return:\n"
           u8"    Total capacitance [pF].\n\n"
           u8"Note:\n"
           u8"    This method can only be used it there are exactly two boundary conditions\n"
           u8"    specifying the voltage. Otherwise use :meth:`get_electrostatic_energy` to\n"
           u8"    obtain the stored energy :math:`W` and compute the capacitance as:\n"
           u8"    :math:`C = 2 \\, W / U^2`, where :math:`U` is the applied voltage.\n"
    );
    METHOD(get_total_heat, getTotalHeat,
           u8"Get the total heat produced by the current flowing in the structure.\n\n"
           u8"Return:\n"
           u8"    Total produced heat [mW].\n"
    );*/
    registerFemSolver(solver);
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(ddm2d)
{
    py_enum<Stat>()
        .value("MAXWELL_BOLTZMANN", STAT_MB)
        .value("FERMI_DIRAC", STAT_FD)
    ;

    py_enum<ContType>()
        .value("OHMIC", OHMIC)
        .value("SCHOTTKY", SCHOTTKY)
    ;

    register_drift_diffusion_solver<Geometry2DCartesian>("DriftDiffusion2D", "Cartesian");

    register_drift_diffusion_solver<Geometry2DCylindrical>("DriftDiffusionCyl", "cylindrical");
}
