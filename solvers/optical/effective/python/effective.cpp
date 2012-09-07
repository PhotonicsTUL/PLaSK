/** \file
 * Python wrapper for optical/effective solvers.
 */
#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
using namespace plask;
using namespace plask::python;

#include "../eim.h"
#include "../efm.h"
using namespace plask::solvers::effective;

static py::object EffectiveIndex2DSolver_getSymmetry(const EffectiveIndex2DSolver& self) {
    switch (self.symmetry) {
        case EffectiveIndex2DSolver::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2DSolver::SYMMETRY_NEGATIVE: return py::object("negative");
        case EffectiveIndex2DSolver::NO_SYMMETRY: return py::object();
    }
    return py::object();
}

static void EffectiveIndex2DSolver_setSymmetry(EffectiveIndex2DSolver& self, py::object symmetry) {
    if (symmetry == py::object()) { self.symmetry = EffectiveIndex2DSolver::NO_SYMMETRY; return; }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none" ) {
            self.symmetry = EffectiveIndex2DSolver::NO_SYMMETRY; return;
        }
        else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            self.symmetry = EffectiveIndex2DSolver::SYMMETRY_POSITIVE; return;
        }
        else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" || sym == "-1") {
            self.symmetry = EffectiveIndex2DSolver::SYMMETRY_NEGATIVE; return;
        }
        throw py::error_already_set();
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym ==  0) { self.symmetry = EffectiveIndex2DSolver::NO_SYMMETRY; return; }
            else if (sym == +1) { self.symmetry = EffectiveIndex2DSolver::SYMMETRY_POSITIVE; return; }
            else if (sym == -1) { self.symmetry = EffectiveIndex2DSolver::SYMMETRY_NEGATIVE; return; }
            throw py::error_already_set();
        } catch (py::error_already_set) {
            throw ValueError("Wrong symmetry specification.");
        }
    }
}

static std::string EffectiveIndex2DSolver_getPolarization(const EffectiveIndex2DSolver& self) {
    return self.polarization==EffectiveIndex2DSolver::TE ? "TE" : "TM";
}

static void EffectiveIndex2DSolver_setPolarization(EffectiveIndex2DSolver& self, std::string polarization) {
    if (polarization == "TE" || polarization == "s" ) {
        self.polarization = EffectiveIndex2DSolver::TE; return;
    }
    if (polarization == "TM" || polarization == "p") {
        self.polarization = EffectiveIndex2DSolver::TM; return;
    }
}

py::object EffectiveIndex2DSolver_getStripeDeterminant(EffectiveIndex2DSolver& self, int stripe, py::object val)
{
    if (!self.getMesh()) self.setSimpleMesh();
    if (stripe < 0) stripe = self.getMesh()->tran().size() + 1 + stripe;
    if (stripe < 0 || size_t(stripe) >= self.getMesh()->tran().size() + 1) throw IndexError("wrong stripe number");

    return UFUNC<dcomplex>([&](dcomplex x){return self.getStripeDeterminant(stripe, x);}, val);
}

py::object EffectiveFrequencyCylSolver_getStripeDeterminant(EffectiveFrequencyCylSolver& self, int stripe, py::object val)
{
    if (!self.getMesh()) self.setSimpleMesh();
    if (stripe < 0) stripe = self.getMesh()->tran().size() + stripe;
    if (stripe < 0 || size_t(stripe) >= self.getMesh()->tran().size()) throw IndexError("wrong stripe number");

    return UFUNC<dcomplex>([&](dcomplex x){return self.getStripeDeterminant(stripe, x);}, val);
}

static py::object EffectiveIndex2DSolver_getDeterminant(EffectiveIndex2DSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}

static py::object EffectiveFrequencyCylSolver_getDeterminant(EffectiveFrequencyCylSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}

static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

dcomplex EffectiveFrequencyCylSolver_getLambda0(const EffectiveFrequencyCylSolver& self) {
    return 2e3*M_PI / self.k0;
}

void EffectiveFrequencyCylSolver_setLambda0(EffectiveFrequencyCylSolver& self, dcomplex lambda0) {
    self.k0 = 2e3*M_PI / lambda0;
}


/**
 * Initialization of your solver to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(effective)
{
    if (!plask_import_array()) throw(py::error_already_set());

    {CLASS(EffectiveIndex2DSolver, "EffectiveIndex2D",
        "Calculate optical modes and optical field distribution using the effective index\n"
        "method in two-dimensional Cartesian space.")
        __solver__.add_property("symmetry", &EffectiveIndex2DSolver_getSymmetry, &EffectiveIndex2DSolver_setSymmetry, "Symmetry of the searched modes");
        __solver__.add_property("polarization", &EffectiveIndex2DSolver_getPolarization, &EffectiveIndex2DSolver_setPolarization, "Polarization of the searched modes");
        RW_FIELD(outer_distance, "Distance outside outer borders where material is sampled");
        RO_FIELD(root, "Configuration of the global rootdigger");
        RO_FIELD(striperoot, "Configuration of the rootdigger for a single stripe");
        METHOD(setSimpleMesh, "Set simple mesh based on the geometry elements bounding boxes");
        METHOD(setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry elements bounding boxes", "points");
        METHOD(computeMode, "Find the mode near the specified effective index", "neff");
        METHOD(findModes, "Find the modes within the specified range", "start", "end", arg("steps")=100, arg("nummodes")=99999999);
        METHOD(findModesMap, "Find approximate modes by scanning the desired range.\nValues returned by this method can be provided to computeMode to get the full solution.", "start", "end", arg("steps")=100);
        METHOD(setMode, "Set the current mode the specified effective index.\nneff can be a value returned e.g. by findModes.", "neff");
        __solver__.def("getStripeDeterminant", &EffectiveIndex2DSolver_getStripeDeterminant, "Get single stripe modal determinant for debugging purposes",
                       (py::arg("stripe"), "neff"));
        __solver__.def("getDeterminant", &EffectiveIndex2DSolver_getDeterminant, "Get modal determinant for debugging purposes", (py::arg("neff")));
        RECEIVER(inWavelength, "Wavelength of the light");
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        RECEIVER(inGain, "Optical gain in the active region");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
    }

    {CLASS(EffectiveFrequencyCylSolver, "EffectiveFrequencyCyl",
        "Calculate optical modes and optical field distribution using the effective frequency\n"
        "method in two-dimensional cylindrical space.")
        RW_FIELD(l, "Radial mode number");
        RW_FIELD(k0, "Reference normalized frequency");
        __solver__.add_property("lam0", &EffectiveFrequencyCylSolver_getLambda0, &EffectiveFrequencyCylSolver_getLambda0, "Reference wavelength");
        RW_FIELD(outer_distance, "Distance outside outer borders where material is sampled");
        RO_FIELD(root, "Configuration of the global rootdigger");
        RO_FIELD(striperoot, "Configuration of the rootdigger for a single stripe");
        METHOD(setSimpleMesh, "Set simple mesh based on the geometry elements bounding boxes");
        METHOD(setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry elements bounding boxes", "points");
        METHOD(computeMode, "Find the mode near the specified effective index", "wavelength");
        METHOD(findModes, "Find the modes within the specified range", "start", "end", arg("steps")=100, arg("nummodes")=99999999);
        METHOD(findModesMap, "Find approximate modes by scanning the desired range.\nValues returned by this method can be provided to computeMode to get the full solution.", "start", "end", arg("steps")=100);
        __solver__.def("getStripeDeterminant", &EffectiveFrequencyCylSolver_getStripeDeterminant, "Get single stripe modal determinant for debugging purposes",
                       (py::arg("stripe"), "veff"));
        __solver__.def("getDeterminant", &EffectiveFrequencyCylSolver_getDeterminant, "Get modal determinant for debugging purposes", py::arg("v"));
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        RECEIVER(inGain, "Optical gain in the active region");
        PROVIDER(outWavelength, "Wavelength of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
    }

    py::class_<RootDigger::Params, boost::noncopyable>("RootdiggerParams", py::no_init)
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration")
        .def_readwrite("maxiterations", &RootDigger::Params::maxiterations, "Maximum number of iterations")
    ;
}
