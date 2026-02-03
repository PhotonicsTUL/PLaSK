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

#include "../capacitance2d.hpp"
using namespace plask::electrical::capacitance;

template <typename __Class__>
inline static ExportSolver<__Class__> register_capacitance_solver(const char* name, const char* geoname) {
    ExportSolver<__Class__> solver(name,
                                   format(u8"{0}(name=\"\")\n\n"
                                          u8"Finite element AC electric solver for {1} geometry.",
                                          name, geoname)
                                       .c_str(),
                                   py::init<std::string>(py::arg("name") = ""));
    METHOD(compute, compute, u8"Run calculations");
    METHOD(get_active_current, getActiveCurrent, u8"Get total current flowing through active region (mA)", py::arg("nact") = 0);
    RECEIVER(inTemperature, u8"");
    RECEIVER(inDifferentialConductivity, u8"");
    PROVIDER(outAcVoltage, u8"");
    PROVIDER(outAcCurrentDensity, u8"");
    BOUNDARY_CONDITIONS(voltage_boundary, u8"Boundary conditions of the first kind (constant potential)");
    RW_PROPERTY(frequency, getFrequency, setFrequency, u8"AC modulation frequency (MHz)");
    RO_PROPERTY(S11, getS11, u8"Scattering parameter <i>S</i><sub>11</sub> at the current frequency");

    // solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
    solver.add_property("empty_elements", &__Class__::getEmptyElements, &__Class__::setEmptyElements,
                        "Should empty regions (e.g. air) be included into computation domain?");


    return solver;
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(capacitance) {
    register_capacitance_solver<Capacitance2DSolver<Geometry2DCartesian>>("Capacitance2D", "2D Cartesian");
    register_capacitance_solver<Capacitance2DSolver<Geometry2DCylindrical>>("CapacitanceCyl", "2D cylindrical");
}
