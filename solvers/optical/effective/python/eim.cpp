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
#include "../eim.hpp"

namespace plask { namespace optical { namespace effective { namespace python {


static py::object EffectiveIndex2D_getSymmetry(const EffectiveIndex2D::Mode& self) {
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: return py::object("negative");
        default: return py::object();
    }
    return py::object();
}

static EffectiveIndex2D::Symmetry parseSymmetry(py::object symmetry) {
    if (symmetry.is_none()) {
        return EffectiveIndex2D::SYMMETRY_DEFAULT;
    }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none") {
            return EffectiveIndex2D::SYMMETRY_NONE;
        } else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            return EffectiveIndex2D::SYMMETRY_POSITIVE;
        } else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" ||
                   sym == "-1") {
            return EffectiveIndex2D::SYMMETRY_NEGATIVE;
        }
        throw py::error_already_set();
    } catch (py::error_already_set&) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym == 0) {
                return EffectiveIndex2D::SYMMETRY_NONE;
            } else if (sym == +1) {
                return EffectiveIndex2D::SYMMETRY_POSITIVE;
            } else if (sym == -1) {
                return EffectiveIndex2D::SYMMETRY_NEGATIVE;
            }
            throw py::error_already_set();
        } catch (py::error_already_set&) {
            throw ValueError(u8"Wrong symmetry specification.");
        }
    }
}

static std::string EffectiveIndex2D_getPolarization(const EffectiveIndex2D& self) {
    return self.getPolarization() == EffectiveIndex2D::TE ? "TE" : "TM";
}

static void EffectiveIndex2D_setPolarization(EffectiveIndex2D& self, std::string polarization) {
    if (polarization == "TE" || polarization == "s") {
        self.setPolarization(EffectiveIndex2D::TE);
        return;
    }
    if (polarization == "TM" || polarization == "p") {
        self.setPolarization(EffectiveIndex2D::TM);
        return;
    }
}

static py::object EffectiveIndex2D_getDeterminant(EffectiveIndex2D& self, py::object val) {
    return UFUNC<dcomplex>([&](dcomplex x) { return self.getDeterminant(x); }, val, "EffectiveIndex2D.get_determinant", "neff");
}
py::object EffectiveIndex2D_getVertDeterminant(EffectiveIndex2D& self, py::object val) {
    return UFUNC<dcomplex>([&](dcomplex x) { return self.getVertDeterminant(x); }, val, "EffectiveIndex2D.get_vert_determinant",
                           "neff");
}

py::object EffectiveIndex2D_getMirrors(const EffectiveIndex2D& self) {
    if (!self.mirrors) return py::object();
    return py::make_tuple(self.mirrors->first, self.mirrors->second);
}

void EffectiveIndex2D_setMirrors(EffectiveIndex2D& self, py::object value) {
    if (value.is_none())
        self.mirrors.reset();
    else {
        try {
            double v = py::extract<double>(value);
            self.mirrors.reset(std::make_pair(v, v));
        } catch (py::error_already_set&) {
            PyErr_Clear();
            try {
                if (py::len(value) != 2) throw py::error_already_set();
                self.mirrors.reset(
                    std::make_pair<double, double>(double(py::extract<double>(value[0])), double(py::extract<double>(value[1]))));
            } catch (py::error_already_set&) {
                throw ValueError(u8"None, float, or tuple of two floats required");
            }
        }
    }
}

static size_t EffectiveIndex2D_findMode(EffectiveIndex2D& self, py::object neff, py::object symmetry) {
    return self.findMode(py::extract<dcomplex>(neff), parseSymmetry(symmetry));
}

std::vector<size_t> EffectiveIndex2D_findModes(EffectiveIndex2D& self,
                                               dcomplex neff1,
                                               dcomplex neff2,
                                               py::object symmetry,
                                               size_t resteps,
                                               size_t imsteps,
                                               dcomplex eps) {
    return self.findModes(neff1, neff2, parseSymmetry(symmetry), resteps, imsteps, eps);
}

static size_t EffectiveIndex2D_setMode(EffectiveIndex2D& self, py::object neff, py::object symmetry) {
    return self.setMode(py::extract<dcomplex>(neff), parseSymmetry(symmetry));
}

std::string EffectiveIndex2D_Mode_str(const EffectiveIndex2D::Mode& self) {
    std::string sym;
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: sym = "positive"; break;
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: sym = "negative"; break;
        default: sym = "none";
    }
    return format(u8"<neff: {:.3f}{:+.3g}j, symmetry: {}, power: {:.2g}mW>", real(self.neff), imag(self.neff), sym, self.power);
}
std::string EffectiveIndex2D_Mode_repr(const EffectiveIndex2D::Mode& self) {
    std::string sym;
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: sym = "'positive'"; break;
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: sym = "'negative'"; break;
        default: sym = "None";
    }
    return format(u8"EffectiveIndex2D.Mode(neff={0}, symmetry={1}, power={2})", str(self.neff), sym, self.power);
}

struct EffectiveIndex2D_getDeltaNeff_Name {
    static const char* const val;
};

const char* const EffectiveIndex2D_getDeltaNeff_Name::val = "EffectiveIndex2D.get_delta_neff";

/**
 * Initialization of your solver to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */

void registerEffectiveIndex2D()
{
    if (!plask_import_array()) throw(py::error_already_set());

    CLASS(EffectiveIndex2D, "EffectiveIndex2D",
            u8"Calculate optical modes and optical field distribution using the effective index\n"
            u8"method in two-dimensional Cartesian space.\n")
    solver.add_property("mesh", &EffectiveIndex2D::getMesh, &Optical_setMesh<EffectiveIndex2D>,
                        u8"Mesh provided to the solver.");
    solver.add_property("polarization", &EffectiveIndex2D_getPolarization, &EffectiveIndex2D_setPolarization,
                        u8"Polarization of the searched modes.");
    RO_FIELD(root,
                u8"Configuration of the root searching algorithm for horizontal component of the\n"
                u8"mode.\n\n" ROOTDIGGER_ATTRS_DOC);
    RO_FIELD(stripe_root,
                u8"Configuration of the root searching algorithm for vertical component of the mode\n"
                u8"in a single stripe.\n\n" ROOTDIGGER_ATTRS_DOC);
    RW_FIELD(emission, u8"Emission direction.");
    METHOD(set_simple_mesh, setSimpleMesh, u8"Set simple mesh based on the geometry objects bounding boxes.");
    // METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the
    // geometry objects bounding boxes", "points");
    METHOD(search_vneff, searchVNeffs,
            u8"Find the effective indices in the vertical direction within the specified range\n"
            u8"using global method.\n\n"
            u8"Args:\n" SEARCH_ARGS_DOC
            "\n"
            u8"Returns:\n"
            u8"    list of floats: List of the found effective indices in the vertical\n"
            u8"    direction.\n",
            arg("start") = 0., arg("end") = 0., arg("resteps") = 256, arg("imsteps") = 64, arg("eps") = dcomplex(1e-6, 1e-9));
    solver.def("find_mode", &EffectiveIndex2D_findMode,
                u8"Compute the mode near the specified effective index.\n\n"
                u8"Args:\n"
                u8"    neff (complex): Starting point of the root search.\n"
                u8"    symmetry ('+' or '-'): Symmetry of the mode to search. If this parameter\n"
                u8"                           is not specified, the default symmetry is used:\n"
                u8"                           positive mode symmetry fir symmetrical geometries\n"
                u8"                           and no symmetry for asymmetrical geometries.\n\n"
                u8"Returns:\n"
                u8"    integer: Index in the :attr:`modes` list of the found mode.\n",
                (arg("neff"), arg("symmetry") = py::object()));
    solver.def("find_modes", &EffectiveIndex2D_findModes,
                u8"Find the modes within the specified range using global method.\n\n"
                u8"Args:\n" SEARCH_ARGS_DOC
                "\n"
                u8"Returns:\n"
                u8"    list of integers: List of the indices in the :attr:`modes` list of the found\n"
                u8"    modes.\n",
                (arg("start") = 0., arg("end") = 0., arg("symmetry") = py::object(), arg("resteps") = 256, arg("imsteps") = 64,
                arg("eps") = dcomplex(1e-6, 1e-9)));
    solver.def("set_mode", EffectiveIndex2D_setMode,
                u8"Set the current mode the specified effective index.\n\n"
                u8"Args:\n"
                u8"    neff (complex): Mode effective index.\n"
                u8"    symmetry ('+' or '-'): Symmetry of the mode to search.\n",
                "neff", arg("symmetry") = py::object());
    METHOD(clear_modes, clearModes, "Clear all computed modes.\n");
    solver.def("get_total_absorption", (double (EffectiveIndex2D::*)(size_t)) & EffectiveIndex2D::getTotalAbsorption,
                u8"Get total energy absorbed by from a mode in unit time.\n\n"
                u8"Args:\n"
                u8"    num (int): number of the mode.\n\n"
                u8"Returns:\n"
                u8"    Total absorbed energy [mW].\n",
                py::arg("num") = 0);
    RW_PROPERTY(vat, getStripeX, setStripeX, u8"Horizontal position of the main stripe (with dominant mode).");
    RW_FIELD(vneff, u8"Effective index in the vertical direction.");
    solver.def("get_delta_neff", &getDeltaNeff<EffectiveIndex2D,EffectiveIndex2D_getDeltaNeff_Name>, py::arg("pos"),
                u8"Return effective index part for lateral propagation at specified horizontal\n"
                u8"position.\n\n"
                u8"Args:\n"
                u8"    pos (float or array of floats): Horizontal position to get the effective\n"
                u8"                                    index.\n");
    solver.add_property("mirrors", EffectiveIndex2D_getMirrors, EffectiveIndex2D_setMirrors,
                        u8"Mirror reflectivities. If None then they are automatically estimated from the"
                        u8"Fresnel equations.\n");
    solver.def(u8"get_vert_determinant", EffectiveIndex2D_getVertDeterminant,
                u8"Get vertical modal determinant for debugging purposes.\n\n"
                u8"Args:\n"
                u8"    neff (complex of numeric array of complex): Vertical effective index value\n"
                u8"    to compute the determinant at.\n\n"
                u8"Returns:\n"
                u8"    complex or list of complex: Determinant at the vertical effective index\n"
                u8"    *neff* or an array matching its size.",
                py::arg("neff"));
    solver.def("get_determinant", &EffectiveIndex2D_getDeterminant,
                u8"Get modal determinant.\n\n"
                u8"Args:\n"
                u8"    neff (complex or array of complex): effective index value\n"
                u8"    to compute the determinant at.\n\n"
                u8"Returns:\n"
                u8"    complex or list of complex: Determinant at the effective index *neff* or\n"
                u8"    an array matching its size.",
                py::arg("neff"));
    RW_PROPERTY(wavelength, getWavelength, setWavelength, "Current wavelength.");
    RECEIVER(inTemperature, "");
    RECEIVER(inGain, "");
    RECEIVER(inCarriersConcentration, "");
    PROVIDER(outNeff, "");
    PROVIDER(outLightMagnitude, "");
    PROVIDER(outLightE, "");
    solver.def_readonly(
        "outElectricField",
        reinterpret_cast<ProviderFor<LightE, Geometry2DCartesian> EffectiveIndex2D::*>(&EffectiveIndex2D::outLightE),
        "Alias for :attr:`outLightE`.");
    PROVIDER(outRefractiveIndex, "");
    PROVIDER(outHeat, "");
    RO_FIELD(modes,
                u8"List of the computed modes.\n\n"
                u8".. rubric:: Item Attributes\n\n"
                u8".. autosummary::\n\n"
                u8"   ~optical.effective.EffectiveIndex2D.Mode.neff\n"
                u8"   ~optical.effective.EffectiveIndex2D.Mode.symmetry\n"
                u8"   ~optical.effective.EffectiveIndex2D.Mode.power\n"
                u8"   ~optical.effective.EffectiveIndex2D.Mode.total_absorption\n\n"
                u8":rtype: optical.effecticve.EffectiveIndex2D.Mode\n");

    py::scope scope = solver;
    (void)scope;  // don't warn about unused variable scope

    register_vector_of<EffectiveIndex2D::Mode>("Modes");

    py::class_<EffectiveIndex2D::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .def_readonly("neff", &EffectiveIndex2D::Mode::neff, u8"Mode effective index.")
        .add_property("symmetry", &EffectiveIndex2D_getSymmetry, u8"Mode symmetry ('positive', 'negative', or None).")
        .def_readwrite("power", &EffectiveIndex2D::Mode::power, u8"Total power emitted into the mode [mW].")
        .add_property("loss", &EffectiveIndex2D::Mode::loss, u8"Mode losses [1/cm].")
        .add_property("total_absorption", &Mode_total_absorption<EffectiveIndex2D>,
                        u8"Cumulated absorption for the mode [mW].\n\n"
                        u8"This property combines gain in active region and absorption in the whole\n"
                        u8"structure.")
        .def("__str__", &EffectiveIndex2D_Mode_str)
        .def("__repr__", &EffectiveIndex2D_Mode_repr);

    py_enum<EffectiveIndex2D::Emission>().value("FRONT", EffectiveIndex2D::FRONT).value("BACK", EffectiveIndex2D::BACK);
}

}}}}  // namespace plask::optical::effective::python
