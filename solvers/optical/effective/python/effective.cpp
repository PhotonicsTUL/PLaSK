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

#define ROOTDIGGER_ATTRS_DOC \
    ".. rubric:: Attributes\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.effective.RootParams.maxiter\n" \
    "   ~optical.effective.RootParams.tolf_max\n" \
    "   ~optical.effective.RootParams.tolf_min\n" \
    "   ~optical.effective.RootParams.tolx\n"

#define SEARCH_ARGS_DOC \
    "Args:\n" \
    "    start (complex): Start of the search range (0 means automatic).\n" \
    "    end (complex): End of the search range (0 means automatic).\n" \
    "    resteps (integer): Number of steps on the real axis during the search.\n" \
    "    imsteps (integer): Number of steps on the imaginary axis during the search.\n" \
    "    eps (complex): required precision of the search.\n" \

static py::object EffectiveIndex2DSolver_getSymmetry(const EffectiveIndex2DSolver::Mode& self) {
    switch (self.symmetry) {
        case EffectiveIndex2DSolver::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2DSolver::SYMMETRY_NEGATIVE: return py::object("negative");
        default: return py::object();
    }
    return py::object();
}

static EffectiveIndex2DSolver::Symmetry parseSymmetry(py::object symmetry) {
    if (symmetry == py::object()) { return EffectiveIndex2DSolver::SYMMETRY_DEFAULT; }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none" ) {
            return EffectiveIndex2DSolver::SYMMETRY_NONE;
        }
        else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            return EffectiveIndex2DSolver::SYMMETRY_POSITIVE;
        }
        else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" || sym == "-1") {
            return EffectiveIndex2DSolver::SYMMETRY_NEGATIVE;
        }
        throw py::error_already_set();
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym ==  0) { return EffectiveIndex2DSolver::SYMMETRY_NONE; }
            else if (sym == +1) { return EffectiveIndex2DSolver::SYMMETRY_POSITIVE; }
            else if (sym == -1) { return EffectiveIndex2DSolver::SYMMETRY_NEGATIVE; }
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

static py::object EffectiveIndex2DSolver_getDeterminant(EffectiveIndex2DSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}
py::object EffectiveIndex2DSolver_getVertDeterminant(EffectiveIndex2DSolver& self, py::object val)
{
    return UFUNC<dcomplex>([&](dcomplex x){return self.getVertDeterminant(x);}, val);
}


static py::object EffectiveFrequencyCylSolver_getDeterminant(EffectiveFrequencyCylSolver& self, py::object val, int m)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x, m);}, val);
}
static py::object EffectiveFrequencyCylSolver_getVertDeterminant(EffectiveFrequencyCylSolver& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getVertDeterminant(x);}, val);
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

size_t EffectiveIndex2DSolver_findMode(EffectiveIndex2DSolver& self, py::object neff, py::object symmetry) {
    py::extract<dcomplex> neff_as_complex(neff);
    if (neff_as_complex.check())
        return self.findMode(neff_as_complex, parseSymmetry(symmetry));
    else {
        if (py::len(neff) != 2) throw TypeError("'neff' must be either complex or sequence of two complex");
        return self.findMode(py::extract<dcomplex>(neff[0]), py::extract<dcomplex>(neff[1]), parseSymmetry(symmetry));
    }
}

std::vector<size_t> EffectiveIndex2DSolver_findModes(EffectiveIndex2DSolver& self, dcomplex neff1, dcomplex neff2, py::object symmetry, size_t resteps, size_t imsteps, dcomplex eps) {
    return self.findModes(neff1, neff2, parseSymmetry(symmetry), resteps, imsteps, eps);
}

size_t EffectiveFrequencyCylSolver_findMode(EffectiveFrequencyCylSolver& self, py::object lam, int m) {
    py::extract<dcomplex> lam_as_complex(lam);
    if (lam_as_complex.check())
        return self.findMode(lam_as_complex, m);
    else {
        if (py::len(lam) != 2) throw TypeError("'lam' must be either complex or sequence of two complex");
        return self.findMode(py::extract<dcomplex>(lam[0]), py::extract<dcomplex>(lam[1]), m);
    }
}

double EffectiveFrequencyCylSolver_Mode_Wavelength(const EffectiveFrequencyCylSolver::Mode& mode) {
    return real(mode.lam);
}

double EffectiveFrequencyCylSolver_Mode_ModalLoss(const EffectiveFrequencyCylSolver::Mode& mode) {
    return imag(2e4 * 2e3*M_PI / mode.lam);
}

template <typename SolverT>
static void Optical_setMesh(SolverT& self, py::object omesh) {
    try {
        shared_ptr<OrderedMesh1D> mesh = py::extract<shared_ptr<OrderedMesh1D>>(omesh);
        self.setHorizontalMesh(mesh);
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            shared_ptr<MeshGeneratorD<1>> meshg = py::extract<shared_ptr<MeshGeneratorD<1>>>(omesh);
            self.setMesh(make_shared<RectilinearMesh2DFrom1DGenerator>(meshg));
        } catch (py::error_already_set) {
            PyErr_Clear();
            plask::python::detail::ExportedSolverDefaultDefs<SolverT>::Solver_setMesh(self, omesh);
        }
    }
}

py::object EffectiveFrequencyCylSolver_getStripeR(const EffectiveFrequencyCylSolver& self) {
    double r = self.getStripeR();
    if (std::isnan(r)) return py::object();
    return py::object(r);
}

void EffectiveFrequencyCylSolver_setStripeR(EffectiveFrequencyCylSolver& self, py::object r) {
    if (r == py::object()) self.useAllStripes();
    else self.setStripeR(py::extract<double>(r));
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
        "method in two-dimensional Cartesian space.\n")
        solver.add_property("mesh", &EffectiveIndex2DSolver::getMesh, &Optical_setMesh<EffectiveIndex2DSolver>, "Mesh provided to the solver.");
        solver.add_property("polarization", &EffectiveIndex2DSolver_getPolarization, &EffectiveIndex2DSolver_setPolarization, "Polarization of the searched modes.");
        RW_FIELD(outdist, "Distance outside outer borders where material is sampled.");
        RO_FIELD(root,
                 "Configuration of the root searching algorithm for horizontal component of the\n"
                 "mode.\n\n"
                 ROOTDIGGER_ATTRS_DOC
                );
        RO_FIELD(stripe_root,
                 "Configuration of the root searching algorithm for vertical component of the mode\n"
                 "in a single stripe.\n\n"
                 ROOTDIGGER_ATTRS_DOC
                );
        RW_FIELD(emission, "Emission direction.");
        METHOD(set_simple_mesh, setSimpleMesh, "Set simple mesh based on the geometry objects bounding boxes.");
        // METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry objects bounding boxes", "points");
        METHOD(search_vneff, searchVNeffs,
               "Find the effective indices in the vertical direction within the specified range\n"
               "using global method.\n\n"
               SEARCH_ARGS_DOC"\n"
               "Returns:\n"
               "    list of floats: List of the found effective indices in the vertical\n"
               "    direction.\n",
               arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        solver.def("find_mode", &EffectiveIndex2DSolver_findMode,
                   "Compute the mode near the specified effective index.\n\n"
                   "Args:\n"
                   "    neff (complex): Starting point of the root search.\n"
                   "    symmetry ('+' or '-'): Symmetry of the mode to search.\n\n"
                   "Returns:\n"
                   "    integer: Index in the :attr:`modes` list of the found mode.\n",
                   (arg("neff"), arg("symmetry")=py::object()));
        solver.def("find_modes", &EffectiveIndex2DSolver_findModes,
                   "Find the modes within the specified range using global method.\n\n"
                   SEARCH_ARGS_DOC"\n"
                   "Returns:\n"
                   "    list of integers: List of the indices in the :attr:`modes` list of the found\n"
                   "    modes.\n",
                   (arg("start")=0., arg("end")=0., arg("symmetry")=py::object(), arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9)));
        METHOD(set_mode, setMode,
               "Set the current mode the specified effective index.\n\n"
               "Args:\n"
               "    neff (complex): Mode effective index.\n"
               "    symmetry ('+' or '-'): Symmetry of the mode to search.\n",
               "neff", arg("symmetry")=py::object());
        solver.def("get_total_absorption", (double (EffectiveIndex2DSolver::*)(size_t))&EffectiveIndex2DSolver::getTotalAbsorption,
               "Get total energy absorbed by from a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total absorbed energy\n",
               py::arg("num")=0);
        RW_PROPERTY(vat, getStripeX, setStripeX, "Horizontal position of the main stripe (with dominant mode).");
        RW_FIELD(vneff, "Effective index in the vertical direction.");
        solver.add_property("mirrors", EffectiveIndex2DSolver_getMirrors, EffectiveIndex2DSolver_setMirrors,
                    "Mirror reflectivities. If None then they are automatically estimated from the"
                    "Fresnel equations.\n");
        solver.def("get_vert_determinant", EffectiveIndex2DSolver_getVertDeterminant,
                   "Get vertical modal determinant for debugging purposes.\n\n"
                   "Args:\n"
                   "    neff (complex of numeric array of complex): Vertical effective index value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the vertical effective index\n"
                   "    *neff* or an array matching its size.",
                   "neff");
        solver.def("get_determinant", &EffectiveIndex2DSolver_getDeterminant,
                   "Get modal determinant.\n\n"
                   "Args:\n"
                   "    neff (complex of numeric array of complex): effective index value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the effective index *neff* or\n"
                   "    an array matching its size.",
                   (py::arg("neff") , py::arg("polarization")=py::object()));
        RW_PROPERTY(wavelength, getWavelength, setWavelength, "");
        RECEIVER(inTemperature, "");
        RECEIVER(inGain, "");
        PROVIDER(outNeff, "");
        PROVIDER(outLightMagnitude, "");
        PROVIDER(outRefractiveIndex, "");
        PROVIDER(outHeat, "");
        RO_FIELD(modes,
                 "List of the computed modes.\n\n"
                 ".. rubric:: Item Attributes\n\n"
                 ".. autosummary::\n\n"
                 "   ~optical.effective.EffectiveIndex2D.Mode.neff\n"
                 "   ~optical.effective.EffectiveIndex2D.Mode.symmetry\n"
                 "   ~optical.effective.EffectiveIndex2D.Mode.power\n");

        py::scope scope = solver;

        register_vector_of<EffectiveIndex2DSolver::Mode>("Modes");

        py::class_<EffectiveIndex2DSolver::Mode>("Mode", "Detailed information about the mode.", py::no_init)
            .def_readonly("neff", &EffectiveIndex2DSolver::Mode::neff, "Mode effective index.")
            .add_property("symmetry", &EffectiveIndex2DSolver_getSymmetry, "Mode symmetry ('positive', 'negative', or None).")
            .def_readwrite("power", &EffectiveIndex2DSolver::Mode::power, "Total power emitted into the mode [mW].")
        ;

        py_enum<EffectiveIndex2DSolver::Emission>()
            .value("FRONT", EffectiveIndex2DSolver::FRONT)
            .value("BACK", EffectiveIndex2DSolver::BACK)
        ;
    }

    {CLASS(EffectiveFrequencyCylSolver, "EffectiveFrequencyCyl",
        "Calculate optical modes and optical field distribution using the effective frequency\n"
        "method in two-dimensional cylindrical space.\n")
        solver.add_property("mesh", &EffectiveFrequencyCylSolver::getMesh, &Optical_setMesh<EffectiveFrequencyCylSolver>, "Mesh provided to the solver.");
        RW_FIELD(k0, "Reference normalized frequency.");
        RW_FIELD(vlam, "'Vertical wavelength' used as a helper for searching vertical modes.");
        solver.add_property("lam0", &EffectiveFrequencyCylSolver_getLambda0, &EffectiveFrequencyCylSolver_setLambda0, "Reference wavelength.");
        RW_FIELD(outdist, "Distance outside outer borders where material is sampled.");
        RO_FIELD(root,
                 "Configuration of the root searching algorithm for horizontal component of the\n"
                 "mode.\n\n"
                 ROOTDIGGER_ATTRS_DOC
                );
        RO_FIELD(stripe_root,
                 "Configuration of the root searching algorithm for vertical component of the mode\n"
                 "in a single stripe.\n\n"
                 ROOTDIGGER_ATTRS_DOC
                );
        RW_PROPERTY(emission, getEmission, setEmission, "Emission direction.");
        METHOD(set_simple_mesh, setSimpleMesh, "Set simple mesh based on the geometry objects bounding boxes.");
        // METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry objects bounding boxes", "points");
        solver.def("find_mode", &EffectiveFrequencyCylSolver_findMode,
                   "Compute the mode near the specified wavelength.\n\n"
                   "Args:\n"
                   "    lam (complex): Initial wavelength to for root finging algorithm.\n"
                   "    m (integer): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n"
                   "Returns:\n"
                   "    integer: Index in the :attr:`modes` list of the found mode.\n",
                   (arg("lam"), arg("m")=0));
        METHOD(find_modes, findModes,
               "Find the modes within the specified range using global method.\n\n"
               SEARCH_ARGS_DOC"\n"
               "Returns:\n"
               "    list of integers: List of the indices in the :attr:`modes` list of the found\n"
               "    modes.\n",
               arg("start")=0., arg("end")=0., arg("m")=0, arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        solver.def("get_vert_determinant", &EffectiveFrequencyCylSolver_getVertDeterminant,
                   "Get vertical modal determinant for debugging purposes.\n\n"
                   "Args:\n"
                   "    vlam (complex of numeric array of complex): Vertical wavelength value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the vertical wavelength *vlam* or\n"
                   "    an array matching its size.\n",
                   py::arg("vlam"));
        solver.def("get_determinant", &EffectiveFrequencyCylSolver_getDeterminant,
                   "Get modal determinant.\n\n"
                   "Args:\n"
                   "    lam (complex of numeric array of complex): wavelength to compute the\n"
                   "    determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the effective index *neff* or\n"
                   "    an array matching its size.\n",
                   (py::arg("lam"), py::arg("m")=0));
        solver.def("set_mode", (size_t (EffectiveFrequencyCylSolver::*)(dcomplex,int))&EffectiveFrequencyCylSolver::setMode, (py::arg("lam"), py::arg("m")=0));
        solver.def("set_mode", (size_t (EffectiveFrequencyCylSolver::*)(double,double,int))&EffectiveFrequencyCylSolver::setMode,
                   "Set the current mode the specified wavelength.\n\n"
                   "Args:\n"
                   "    lam (float of complex): Mode wavelength.\n"
                   "    loss (float): Mode losses. Allowed only if *lam* is a float.\n"
                   "    m (integer): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n",
                   (py::arg("lam"), "loss", py::arg("m")=0));
        solver.def("get_total_absorption", (double (EffectiveFrequencyCylSolver::*)(size_t))&EffectiveFrequencyCylSolver::getTotalAbsorption,
               "Get total energy absorbed from a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total absorbed energy\n",
               py::arg("num")=0);
        solver.def("get_gain_integral", (double (EffectiveFrequencyCylSolver::*)(size_t))&EffectiveFrequencyCylSolver::getGainIntegral,
               "Get total energy generated in the gain region to a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total absorbed energy\n",
               py::arg("num")=0);
        RECEIVER(inTemperature, "");
        RECEIVER(inGain, "");
        PROVIDER(outWavelength, "");
        PROVIDER(outLoss, "");
        PROVIDER(outLightMagnitude, "");
        PROVIDER(outRefractiveIndex, "");
        PROVIDER(outHeat, "");
        RO_FIELD(modes,
                 "List of the computed modes.\n\n"
                 ".. rubric:: Item Attributes\n\n"
                 ".. autosummary::\n\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.m\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.lam\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.wavelength\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.loss\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.power\n");
        solver.add_property("vat", &EffectiveFrequencyCylSolver_getStripeR, &EffectiveFrequencyCylSolver_setStripeR,
                            "Radial position of at which the vertical part of the field is calculated.\n\n"
                            "Should be a float number or ``None`` to compute effective frequencies for all\n"
                            "the stripes.\n");

        py::scope scope = solver;

        register_vector_of<EffectiveFrequencyCylSolver::Mode>("Modes");

        py::class_<EffectiveFrequencyCylSolver::Mode>("Mode", "Detailed information about the mode.", py::no_init)
            .def_readonly("m", &EffectiveFrequencyCylSolver::Mode::m, "LP_mn mode parameter describing angular dependence.")
            .add_property("lam", &EffectiveFrequencyCylSolver_Mode_Wavelength, "Alias for :attr:`~optical.effective.EffectiveFrequencyCyl.Mode.wavelength`.")
            .add_property("wavelength", &EffectiveFrequencyCylSolver_Mode_Wavelength, "Mode wavelength [nm].")
            .add_property("loss", &EffectiveFrequencyCylSolver_Mode_ModalLoss, "Mode loss [1/cm].")
            .def_readwrite("power", &EffectiveFrequencyCylSolver::Mode::power, "Total power emitted into the mode.")
        ;

        py_enum<EffectiveFrequencyCylSolver::Emission>()
            .value("TOP", EffectiveFrequencyCylSolver::TOP)
            .value("BOTTOM", EffectiveFrequencyCylSolver::BOTTOM)
        ;
    }

    py::class_<RootMuller::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("tolx", &RootMuller::Params::tolx, "Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootMuller::Params::tolf_min, "Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootMuller::Params::tolf_max, "Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootMuller::Params::maxiter, "Maximum number of iterations.")
    ;

    // py::class_<RootBroyden::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
    //     .def_readwrite("tolx", &RootBroyden::Params::tolx, "Absolute tolerance on the argument.")
    //     .def_readwrite("tolf_min", &RootBroyden::Params::tolf_min, "Sufficient tolerance on the function value.")
    //     .def_readwrite("tolf_max", &RootBroyden::Params::tolf_max, "Required tolerance on the function value.")
    //     .def_readwrite("maxstep", &RootBroyden::Params::maxstep, "Maximum step in one iteration.")
    //     .def_readwrite("maxiter", &RootBroyden::Params::maxiter, "Maximum number of iterations.")
    // ;
}
