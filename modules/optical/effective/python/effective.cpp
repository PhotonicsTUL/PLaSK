/** \file
 * Python wrapper for optical/effective modules.
 */
#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../eim.h"
using namespace plask::modules::eim;

py::object EffectiveIndex2dModule_getSymmetry(const EffectiveIndex2dModule& self) {
    switch (self.symmetry) {
        case EffectiveIndex2dModule::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2dModule::SYMMETRY_NEGATIVE: return py::object("negative");
        case EffectiveIndex2dModule::NO_SYMMETRY: return py::object();
    }
    return py::object();
}

void EffectiveIndex2dModule_setSymmetry(EffectiveIndex2dModule& self, py::object symmetry) {
    if (symmetry == py::object()) { self.symmetry = EffectiveIndex2dModule::NO_SYMMETRY; return; }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none" ) {
            self.symmetry = EffectiveIndex2dModule::NO_SYMMETRY; return;
        }
        else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            self.symmetry = EffectiveIndex2dModule::SYMMETRY_POSITIVE; return;
        }
        else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" || sym == "-1") {
            self.symmetry = EffectiveIndex2dModule::SYMMETRY_NEGATIVE; return;
        }
        throw py::error_already_set();
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym ==  0) { self.symmetry = EffectiveIndex2dModule::NO_SYMMETRY; return; }
            else if (sym == +1) { self.symmetry = EffectiveIndex2dModule::SYMMETRY_POSITIVE; return; }
            else if (sym == -1) { self.symmetry = EffectiveIndex2dModule::SYMMETRY_NEGATIVE; return; }
            throw py::error_already_set();
        } catch (py::error_already_set) {
            throw ValueError("wrong symmetry specification");
        }
    }
}

std::string EffectiveIndex2dModule_getPolarization(const EffectiveIndex2dModule& self) {
    return self.polarization==EffectiveIndex2dModule::TE ? "TE" : "TM";
}

void EffectiveIndex2dModule_setPolarization(EffectiveIndex2dModule& self, std::string polarization) {
    if (polarization == "TE" || polarization == "s" ) {
        self.polarization = EffectiveIndex2dModule::TE; return;
    }
    if (polarization == "TM" || polarization == "p") {
        self.polarization = EffectiveIndex2dModule::TM; return;
    }
}


/**
 * Initialization of your module to Python
 *
 * The \a module_name should be changed to match the name of the directory with our module
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(effective)
{
    {CLASS(EffectiveIndex2dModule, "EffectiveIndex2D",
        "Calculate optical modes and optical field distribution using the effective index\n"
        "method in Cartesian two-dimensional space.")
        __module__.add_property("symmetry", &EffectiveIndex2dModule_getSymmetry, &EffectiveIndex2dModule_setSymmetry, "Symmetry of the searched modes");
        __module__.add_property("polarization", &EffectiveIndex2dModule_getPolarization, &EffectiveIndex2dModule_setPolarization, "Polarization of the searched modes");
        RW_FIELD(outer_distance, "Distance outside outer borders where material is sampled");
        RW_FIELD(tolx, "Absolute tolerance on the argument");
        RW_FIELD(tolf_min, "Sufficient tolerance on the function value");
        RW_FIELD(tolf_max, "Required tolerance on the function value");
        RW_FIELD(maxstep, "Maximum step in one iteration");
        RW_FIELD(maxiterations, "Maximum number of iterations");
        METHOD(setSimpleMesh, "Set simple mesh based on the geometry elements bounding boxes");
        METHOD(setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry elements bounding boxes", "points");
        METHOD(computeMode, "Find the mode near the specified effective index", "neff");
        METHOD(findModes, "Find the modes within the specified range", "start", "end", arg("steps")=100, arg("nummodes")=99999999);
        METHOD(findModesMap, "Find approximate modes by scanning the desired range.\nValues returned by this method can be provided to computeMode to get the full solution.", "start", "end", arg("steps")=100);
        RECEIVER(inWavelength, "Wavelength of the light");
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
    }
}
