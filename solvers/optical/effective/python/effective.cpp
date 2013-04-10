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
    switch (self.getSymmetry()) {
        case EffectiveIndex2DSolver::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2DSolver::SYMMETRY_NEGATIVE: return py::object("negative");
        case EffectiveIndex2DSolver::NO_SYMMETRY: return py::object();
    }
    return py::object();
}

static void EffectiveIndex2DSolver_setSymmetry(EffectiveIndex2DSolver& self, py::object symmetry) {
    if (symmetry == py::object()) { self.setSymmetry(EffectiveIndex2DSolver::NO_SYMMETRY); return; }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none" ) {
            self.setSymmetry(EffectiveIndex2DSolver::NO_SYMMETRY); return;
        }
        else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            self.setSymmetry(EffectiveIndex2DSolver::SYMMETRY_POSITIVE); return;
        }
        else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" || sym == "-1") {
            self.setSymmetry(EffectiveIndex2DSolver::SYMMETRY_NEGATIVE); return;
        }
        throw py::error_already_set();
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym ==  0) { self.setSymmetry(EffectiveIndex2DSolver::NO_SYMMETRY); return; }
            else if (sym == +1) { self.setSymmetry(EffectiveIndex2DSolver::SYMMETRY_POSITIVE); return; }
            else if (sym == -1) { self.setSymmetry(EffectiveIndex2DSolver::SYMMETRY_NEGATIVE); return; }
            throw py::error_already_set();
        } catch (py::error_already_set) {
            throw ValueError("Wrong symmetry specification.");
        }
    }
}

static std::string EffectiveIndex2DSolver_getPolarization(const EffectiveIndex2DSolver& self) {
    return self.getPolarization() == EffectiveIndex2DSolver::TE ? "TE" : "TM";
}

static void EffectiveIndex2DSolver_setPolarization(EffectiveIndex2DSolver& self, std::string polarization) {
    if (polarization == "TE" || polarization == "s" ) {
        self.setPolarization(EffectiveIndex2DSolver::TE); return;
    }
    if (polarization == "TM" || polarization == "p") {
        self.setPolarization(EffectiveIndex2DSolver::TM); return;
    }
}

py::object EffectiveIndex2DSolver_getStripeDeterminant(EffectiveIndex2DSolver& self, int stripe, py::object val)
{
    if (!self.getMesh()) self.setSimpleMesh();
    if (stripe < 0) stripe = self.getMesh()->tran().size() + 1 + stripe;
    if (stripe < 0 || size_t(stripe) >= self.getMesh()->tran().size() + 1) throw IndexError("wrong stripe number");

    return UFUNC<dcomplex>([&](dcomplex x){return self.getStripeDeterminant(stripe, x);}, val);
}

py::object EffectiveFrequencyCylSolver_getStripeDeterminantV(EffectiveFrequencyCylSolver& self, int stripe, py::object val)
{
    if (!self.getMesh()) self.setSimpleMesh();
    if (stripe < 0) stripe = self.getMesh()->tran().size() + stripe;
    if (stripe < 0 || size_t(stripe) >= self.getMesh()->tran().size()) throw IndexError("wrong stripe number");

    return UFUNC<dcomplex>([&](dcomplex x){return self.getStripeDeterminantV(stripe, x);}, val);
}

static py::object EffectiveIndex2DSolver_getDeterminant(EffectiveIndex2DSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}

static py::object EffectiveFrequencyCylSolver_getDeterminant(EffectiveFrequencyCylSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}

static py::object EffectiveFrequencyCylSolver_getDeterminantV(EffectiveFrequencyCylSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminantV(x);}, val);
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

py::object EffectiveIndex2DSolver_getMirrors(const EffectiveIndex2DSolver& self) {
    if (!self.mirrors) return py::object();
    return py::make_tuple(self.mirrors->first, self.mirrors->second);
}

void EffectiveIndex2DSolver_setMirrors(EffectiveIndex2DSolver& self, py::object value) {
    if (value == py::object())
        self.mirrors.reset();
    else {
        try {
            double v = py::extract<double>(value);
            self.mirrors.reset(std::make_pair(v,v));
        } catch (py::error_already_set) {
            PyErr_Clear();
            try {
                if (py::len(value) != 2) throw py::error_already_set();
                self.mirrors.reset(std::make_pair<double,double>(py::extract<double>(value[0]),py::extract<double>(value[1])));
            } catch (py::error_already_set) {
                throw ValueError("None, float, or tuple of two floats required");
            }
        }
    }
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
        solver.add_property("symmetry", &EffectiveIndex2DSolver_getSymmetry, &EffectiveIndex2DSolver_setSymmetry, "Symmetry of the searched modes");
        solver.add_property("polarization", &EffectiveIndex2DSolver_getPolarization, &EffectiveIndex2DSolver_setPolarization, "Polarization of the searched modes");
        RW_FIELD(outdist, "Distance outside outer borders where material is sampled");
        RO_FIELD(root, "Configuration of the global rootdigger");
        RO_FIELD(stripe_root, "Configuration of the rootdigger for a single stripe");
        METHOD(set_simple_mesh, setSimpleMesh, "Set simple mesh based on the geometry objects bounding boxes");
        METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry objects bounding boxes", "points");
        METHOD(find_vneffs, findVeffs, "Find the effective index in the vertical direction within the specified range using global method",
               arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        METHOD(compute, computeMode, "Compute the mode near the specified effective index", "neff");
        METHOD(find_modes, findModes, "Find the modes within the specified range using global method",
               arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        METHOD(set_mode, setMode, "Set the current mode the specified effective index.\nneff can be a value returned e.g. by 'find_modes'.", "neff");
        RW_PROPERTY(stripex, getStripeX, setStripeX, "Horizontal position of the main stripe (with dominat mode)");
        RW_FIELD(vneff, "Effective index in the vertical direction");
        solver.add_property("mirrors", EffectiveIndex2DSolver_getMirrors, EffectiveIndex2DSolver_setMirrors,
                    "Mirror reflectivities. If None then they are automatically estimated from Fresenel equations");
        solver.def("get_stripe_determinant", EffectiveIndex2DSolver_getStripeDeterminant, "Get single stripe modal determinant for debugging purposes",
                       (py::arg("stripe"), "neff"));
        solver.def("get_determinant", &EffectiveIndex2DSolver_getDeterminant, "Get modal determinant", (py::arg("neff")));
        RECEIVER(inWavelength, "Wavelength of the light");
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        RECEIVER(inGain, "Optical gain in the active region");
        PROVIDER(outNeff, "Effective index of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
    }

    {CLASS(EffectiveFrequencyCylSolver, "EffectiveFrequencyCyl",
        "Calculate optical modes and optical field distribution using the effective frequency\n"
        "method in two-dimensional cylindrical space.")
        RW_FIELD(m, "Angular mode number");
        RW_FIELD(k0, "Reference normalized frequency");
        solver.add_property("lam0", &EffectiveFrequencyCylSolver_getLambda0, &EffectiveFrequencyCylSolver_setLambda0, "Reference wavelength");
        RW_FIELD(outdist, "Distance outside outer borders where material is sampled");
        RO_FIELD(root, "Configuration of the global rootdigger");
        RO_FIELD(stripe_root, "Configuration of the rootdigger for a single stripe");
        METHOD(set_simple_mesh, setSimpleMesh, "Set simple mesh based on the geometry objects bounding boxes");
        METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry objects bounding boxes", "points");
        METHOD(compute, computeMode, "Compute the mode near the specified wavelength", "wavelength");
        METHOD(find_modes, findModes, "Find the modes within the specified range using global method",
               arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        solver.def("set_mode", (void (EffectiveFrequencyCylSolver::*)(dcomplex))&EffectiveFrequencyCylSolver::setMode,
                   "Set the current mode the specified wavelength.\nlam can be a value returned e.g. by 'find_modes'.", py::arg("lam"));
        solver.def("set_mode", (void (EffectiveFrequencyCylSolver::*)(double,double))&EffectiveFrequencyCylSolver::setMode,
                   "Set the current mode the specified wavelength.\nlam can be a value returned e.g. by 'find_modes'.", (py::arg("lam"), "ext"));
        solver.def("get_stripe_determinant_v", &EffectiveFrequencyCylSolver_getStripeDeterminantV, "Get single stripe modal determinant for debugging purposes",
                   (py::arg("stripe"), "veff"));
        solver.def("get_determinant_v", &EffectiveFrequencyCylSolver_getDeterminantV, "Get modal determinant for frequency parameter v for debugging purposes",
                   py::arg("v"));
        solver.def("get_determinant", &EffectiveFrequencyCylSolver_getDeterminant, "Get modal determinant", py::arg("lam"));
        RECEIVER(inTemperature, "Temperature distribution in the structure");
        RECEIVER(inGain, "Optical gain distribution in the active region");
        PROVIDER(outWavelength, "Wavelength of the last computed mode");
        PROVIDER(outIntensity, "Light intensity of the last computed mode");
    }

    py::class_<RootDigger::Params, boost::noncopyable>("RootdiggerParams", py::no_init)
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, "Maximum number of iterations")
    ;
}
