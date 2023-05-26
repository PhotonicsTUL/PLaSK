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
/** \file
 * Python wrapper for optical/effective solvers.
 */
#include <cmath>
#include <plask/python.hpp>
#include "plask/python_util/ufunc.hpp"
using namespace plask;
using namespace plask::python;

#include "effective.hpp"
#include "../old_efm.hpp"

namespace plask { namespace optical { namespace effective { namespace python {


static py::object OldEffectiveFrequencyCyl_getDeterminant(OldEffectiveFrequencyCyl& self, py::object val, int m) {
    return UFUNC<dcomplex>([&](dcomplex x) { return self.getDeterminant(x, m); }, val,
                           "OldEffectiveFrequencyCyl.get_determinant", "lam");
}
static py::object OldEffectiveFrequencyCyl_getVertDeterminant(OldEffectiveFrequencyCyl& self, py::object val) {
    return UFUNC<dcomplex>([&](dcomplex x) { return self.getVertDeterminant(x); }, val,
                           "OldEffectiveFrequencyCyl.get_vert_determinant", "lam");
}

dcomplex OldEffectiveFrequencyCyl_getLambda0(const OldEffectiveFrequencyCyl& self) { return 2e3 * PI / self.k0; }

void OldEffectiveFrequencyCyl_setLambda0(OldEffectiveFrequencyCyl& self, dcomplex lambda0) { self.k0 = 2e3 * PI / lambda0; }

static size_t OldEffectiveFrequencyCyl_findMode(OldEffectiveFrequencyCyl& self, py::object lam, int m) {
    return self.findMode(py::extract<dcomplex>(lam), m);
}

double OldEffectiveFrequencyCyl_Mode_ModalLoss(const OldEffectiveFrequencyCyl::Mode& mode) { return imag(2e4 * 2e3 * PI / mode.lam); }

py::object OldEffectiveFrequencyCyl_getStripeR(const OldEffectiveFrequencyCyl& self) {
    double r = self.getStripeR();
    if (std::isnan(r)) return py::object();
    return py::object(r);
}

void OldEffectiveFrequencyCyl_setStripeR(OldEffectiveFrequencyCyl& self, py::object r) {
    if (r.is_none())
        self.useAllStripes();
    else
        self.setStripeR(py::extract<double>(r));
}

std::string OldEffectiveFrequencyCyl_Mode_str(const OldEffectiveFrequencyCyl::Mode& self) {
    return format(u8"<m: {:d}, lam: ({:.3f}{:+.3g}j)nm, power: {:.2g}mW>", self.m, real(self.lam), imag(self.lam), self.power);
}
std::string OldEffectiveFrequencyCyl_Mode_repr(const OldEffectiveFrequencyCyl::Mode& self) {
    return format(u8"OldEffectiveFrequencyCyl.Mode(m={0}, lam={1}, power={2})", self.m, str(self.lam), self.power);
}

static double Mode_gain_integral(OldEffectiveFrequencyCyl::Mode& self) { return self.solver->getGainIntegral(self); }

static py::object OldEffectiveFrequencyCyl_getNNg(OldEffectiveFrequencyCyl& self, py::object pos) {
    return UFUNC<dcomplex, double>([&](double p) { return self.getNNg(p); }, pos, "OldEffectiveFrequencyCyl.get_nng", "pos");
}

struct OldEffectiveFrequencyCyl_getDeltaNeff_Name {
    static const char* const val;
};

const char* const OldEffectiveFrequencyCyl_getDeltaNeff_Name::val = "OldEffectiveFrequencyCyl.get_delta_neff";

/**
 * Initialization of your solver to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
void registerOldEffectiveFrequencyCyl()
{
    if (!plask_import_array()) throw(py::error_already_set());

    CLASS(OldEffectiveFrequencyCyl, "OldEffectiveFrequencyCyl",
            u8"Calculate optical modes and optical field distribution using the effective\n"
            u8"frequency method in two-dimensional cylindrical space.\n")
    solver.add_property("mesh", &OldEffectiveFrequencyCyl::getMesh, &Optical_setMesh<OldEffectiveFrequencyCyl>,
                        "Mesh provided to the solver.");
    RW_FIELD(k0, u8"Reference normalized frequency.");
    RW_FIELD(vlam, u8"'Vertical wavelength' used as a helper for searching vertical modes.");
    solver.add_property("lam0", &OldEffectiveFrequencyCyl_getLambda0, &OldEffectiveFrequencyCyl_setLambda0,
                        u8"Reference wavelength.");
    RO_FIELD(root,
                u8"Configuration of the root searching algorithm for horizontal component of the\n"
                u8"mode.\n\n" ROOTDIGGER_ATTRS_DOC);
    RO_FIELD(stripe_root,
                u8"Configuration of the root searching algorithm for vertical component of the mode\n"
                u8"in a single stripe.\n\n" ROOTDIGGER_ATTRS_DOC);
    //         RW_PROPERTY(asymptotic, getAsymptotic, setAsymptotic,
    //                     "Flag indicating whether the solver uses asymptotic exponential field\n"
    //                     "in the outermost layer.")
    RW_PROPERTY(emission, getEmission, setEmission, u8"Emission direction.");
    solver.def_readwrite("determinant_mode", &OldEffectiveFrequencyCyl::determinant,
                            u8"Radial determinant mode.\n\n"
                            u8"This parameter determines the method used to compute radial determinant.\n"
                            u8"If it is set to 'transfer', 2x2 transfer matrix is used to ensure field\n"
                            u8"continuity at the interfaces. For the 'full' value, one single matrix is\n"
                            u8"constructed for all the interfaces and its determinant is returned.");
    METHOD(set_simple_mesh, setSimpleMesh, u8"Set simple mesh based on the geometry objects bounding boxes.");
    // METHOD(set_horizontal_mesh, setHorizontalMesh, u8"Set custom mesh in horizontal direction, vertical one is based on the
    // geometry objects bounding boxes", "points");
    solver.def("find_mode", &OldEffectiveFrequencyCyl_findMode,
                u8"Compute the mode near the specified wavelength.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Initial wavelength to for root finging algorithm.\n"
                u8"    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n"
                u8"Returns:\n"
                u8"    integer: Index in the :attr:`modes` list of the found mode.\n",
                (arg("lam"), arg("m") = 0));
    METHOD(find_modes, findModes,
            u8"Find the modes within the specified range using global method.\n\n"
            u8"Args:\n"
            u8"    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n" SEARCH_ARGS_DOC
            "\n"
            u8"Returns:\n"
            u8"    list of integers: List of the indices in the :attr:`modes` list of the found\n"
            u8"    modes.\n",
            arg("start") = 0., arg("end") = 0., arg("m") = 0, arg("resteps") = 256, arg("imsteps") = 64,
            arg("eps") = dcomplex(1e-6, 1e-9));
    solver.def("get_vert_determinant", &OldEffectiveFrequencyCyl_getVertDeterminant,
                u8"Get vertical modal determinant for debugging purposes.\n\n"
                u8"Args:\n"
                u8"    vlam (complex of numeric array of complex): Vertical wavelength value\n"
                u8"    to compute the determinant at.\n\n"
                u8"Returns:\n"
                u8"    complex or list of complex: Determinant at the vertical wavelength *vlam* or\n"
                u8"    an array matching its size.\n",
                py::arg("vlam"));
    solver.def("get_determinant", &OldEffectiveFrequencyCyl_getDeterminant,
                u8"Get modal determinant.\n\n"
                u8"Args:\n"
                u8"    lam (complex of numeric array of complex): wavelength to compute the\n"
                u8"                                               determinant at.\n"
                u8"    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n",
                u8"Returns:\n"
                u8"    complex or list of complex: Determinant at the effective index *neff* or\n"
                u8"    an array matching its size.\n",
                (py::arg("lam"), py::arg("m") = 0));
    solver.def("set_mode", (size_t(OldEffectiveFrequencyCyl::*)(dcomplex, int)) & OldEffectiveFrequencyCyl::setMode,
                (py::arg("lam"), py::arg("m") = 0));
    solver.def("set_mode", (size_t(OldEffectiveFrequencyCyl::*)(double, double, int)) & OldEffectiveFrequencyCyl::setMode,
                u8"Set the current mode the specified wavelength.\n\n"
                u8"Args:\n"
                u8"    lam (float of complex): Mode wavelength.\n"
                u8"    loss (float): Mode losses. Allowed only if *lam* is a float.\n"
                u8"    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n",
                (py::arg("lam"), "loss", py::arg("m") = 0));
    METHOD(clear_modes, clearModes, u8"Clear all computed modes.\n");
    solver.def("get_total_absorption", (double (OldEffectiveFrequencyCyl::*)(size_t)) & OldEffectiveFrequencyCyl::getTotalAbsorption,
                u8"Get total energy absorbed from a mode in unit time.\n\n"
                u8"Args:\n"
                u8"    num (int): number of the mode.\n\n"
                u8"Returns:\n"
                u8"    Total absorbed energy [mW].\n",
                py::arg("num") = 0);
    solver.def("get_gain_integral", (double (OldEffectiveFrequencyCyl::*)(size_t)) & OldEffectiveFrequencyCyl::getGainIntegral,
                u8"Get total energy generated in the gain region to a mode in unit time.\n\n"
                u8"Args:\n"
                u8"    num (int): number of the mode.\n\n"
                u8"Returns:\n"
                u8"    Total generated energy [mW].\n",
                py::arg("num") = 0);
    RECEIVER(inTemperature, "");
    RECEIVER(inGain, "");
    RECEIVER(inCarriersConcentration, "");
    PROVIDER(outWavelength, "");
    PROVIDER(outLoss, "");
    PROVIDER(outLightMagnitude, "");
    PROVIDER(outLightE, "");
    solver.def_readonly("outElectricField",
                        reinterpret_cast<ProviderFor<LightE, Geometry2DCylindrical> OldEffectiveFrequencyCyl::*>(
                            &OldEffectiveFrequencyCyl::outLightE),
                        u8"Alias for :attr:`outLightE`.");
    PROVIDER(outRefractiveIndex, "");
    PROVIDER(outHeat, "");
    RO_FIELD(modes,
                u8"List of the computed modes.\n\n"
                u8".. rubric:: Item Attributes\n\n"
                u8".. autosummary::\n\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.m\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.lam\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.wavelength\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.power\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.total_absorption\n"
                u8"   ~optical.effective.OldEffectiveFrequencyCyl.Mode.gain_integral\n\n"
                u8":rtype: optical.effective.EffectiveFrquencyCyl.Mode\n");
    solver.add_property("vat", &OldEffectiveFrequencyCyl_getStripeR, &OldEffectiveFrequencyCyl_setStripeR,
                        u8"Radial position of at which the vertical part of the field is calculated.\n\n"
                        u8"Should be a float number or ``None`` to compute effective frequencies for all\n"
                        u8"the stripes.\n");
    solver.def("get_delta_neff", &getDeltaNeff<OldEffectiveFrequencyCyl,OldEffectiveFrequencyCyl_getDeltaNeff_Name>, py::arg("pos"),
                u8"Return effective index part for lateral propagation at specified radial\n"
                u8"position.\n\n"
                u8"Args:\n"
                u8"    pos (float or array of floats): Radial position to get the effective index.\n");
    solver.def("get_nng", &OldEffectiveFrequencyCyl_getNNg, py::arg("pos"),
                u8"Return average index at specified radial position.\n\n"
                u8"Args:\n"
                u8"    pos (float or array of floats): Radial position to get the effective index.\n");

    py::scope scope = solver;
    (void)scope;  // don't warn about unused variable scope

    register_vector_of<OldEffectiveFrequencyCyl::Mode>("Modes");

    py::class_<OldEffectiveFrequencyCyl::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .def_readonly("m", &OldEffectiveFrequencyCyl::Mode::m, u8"LP_mn mode parameter describing angular dependence.")
        .def_readonly("lam", &OldEffectiveFrequencyCyl::Mode::lam,
                        u8"Alias for :attr:`~optical.effective.OldEffectiveFrequencyCyl.Mode.wavelength`.")
        .def_readonly("wavelength", &OldEffectiveFrequencyCyl::Mode::lam, u8"Mode wavelength [nm].")
        .def_readwrite("power", &OldEffectiveFrequencyCyl::Mode::power, u8"Total power emitted into the mode.")
        .add_property("loss", &OldEffectiveFrequencyCyl::Mode::loss, u8"Mode losses [1/cm].")
        .add_property("total_absorption", &Mode_total_absorption<OldEffectiveFrequencyCyl>,
                        u8"Cumulated absorption for the mode [mW].\n\n"
                        u8"This property combines gain in active region and absorption in the whole\n"
                        u8"structure.")
        .add_property("gain_integral", &Mode_gain_integral, u8"Total gain for the mode [mW].")
        .def("__str__", &OldEffectiveFrequencyCyl_Mode_str)
        .def("__repr__", &OldEffectiveFrequencyCyl_Mode_repr);

    py_enum<OldEffectiveFrequencyCyl::Determinant>()
        .value("OUTWARDS", OldEffectiveFrequencyCyl::DETERMINANT_OUTWARDS)
        .value("INWARDS", OldEffectiveFrequencyCyl::DETERMINANT_INWARDS)
        .value("FULL", OldEffectiveFrequencyCyl::DETERMINANT_FULL)
        .value("TRANSFER", OldEffectiveFrequencyCyl::DETERMINANT_INWARDS);
    py_enum<OldEffectiveFrequencyCyl::Emission>()
        .value("TOP", OldEffectiveFrequencyCyl::TOP)
        .value("BOTTOM", OldEffectiveFrequencyCyl::BOTTOM);
}

}}}}  // namespace plask::optical::effective::python
