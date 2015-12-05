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
    ".. rubric:: Attributes:\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.effective.RootParams.alpha\n" \
    "   ~optical.effective.RootParams.lambd\n" \
    "   ~optical.effective.RootParams.initial_range\n" \
    "   ~optical.effective.RootParams.maxiter\n" \
    "   ~optical.effective.RootParams.maxstep\n" \
    "   ~optical.effective.RootParams.method\n" \
    "   ~optical.effective.RootParams.tolf_max\n" \
    "   ~optical.effective.RootParams.tolf_min\n" \
    "   ~optical.effective.RootParams.tolx\n\n" \
    ":rtype: RootParams\n"

#define SEARCH_ARGS_DOC \
    "    start (complex): Start of the search range (0 means automatic).\n" \
    "    end (complex): End of the search range (0 means automatic).\n" \
    "    resteps (integer): Number of steps on the real axis during the search.\n" \
    "    imsteps (integer): Number of steps on the imaginary axis during the search.\n" \
    "    eps (complex): required precision of the search.\n" \

static py::object EffectiveIndex2D_getSymmetry(const EffectiveIndex2D::Mode& self) {
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: return py::object("positive");
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: return py::object("negative");
        default: return py::object();
    }
    return py::object();
}

static EffectiveIndex2D::Symmetry parseSymmetry(py::object symmetry) {
    if (symmetry == py::object()) { return EffectiveIndex2D::SYMMETRY_DEFAULT; }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "0" || sym == "none" ) {
            return EffectiveIndex2D::SYMMETRY_NONE;
        }
        else if (sym == "positive" || sym == "pos" || sym == "symmeric" || sym == "+" || sym == "+1") {
            return EffectiveIndex2D::SYMMETRY_POSITIVE;
        }
        else if (sym == "negative" || sym == "neg" || sym == "anti-symmeric" || sym == "antisymmeric" || sym == "-" || sym == "-1") {
            return EffectiveIndex2D::SYMMETRY_NEGATIVE;
        }
        throw py::error_already_set();
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            int sym = py::extract<int>(symmetry);
            if (sym ==  0) { return EffectiveIndex2D::SYMMETRY_NONE; }
            else if (sym == +1) { return EffectiveIndex2D::SYMMETRY_POSITIVE; }
            else if (sym == -1) { return EffectiveIndex2D::SYMMETRY_NEGATIVE; }
            throw py::error_already_set();
        } catch (py::error_already_set) {
            throw ValueError("Wrong symmetry specification.");
        }
    }
}

static std::string EffectiveIndex2D_getPolarization(const EffectiveIndex2D& self) {
    return self.getPolarization() == EffectiveIndex2D::TE ? "TE" : "TM";
}

static void EffectiveIndex2D_setPolarization(EffectiveIndex2D& self, std::string polarization) {
    if (polarization == "TE" || polarization == "s" ) {
        self.setPolarization(EffectiveIndex2D::TE); return;
    }
    if (polarization == "TM" || polarization == "p") {
        self.setPolarization(EffectiveIndex2D::TM); return;
    }
}

static py::object EffectiveIndex2D_getDeterminant(EffectiveIndex2D& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x);}, val);
}
py::object EffectiveIndex2D_getVertDeterminant(EffectiveIndex2D& self, py::object val)
{
    return UFUNC<dcomplex>([&](dcomplex x){return self.getVertDeterminant(x);}, val);
}


static py::object EffectiveFrequencyCyl_getDeterminant(EffectiveFrequencyCyl& self, py::object val, int m)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getDeterminant(x, m);}, val);
}
static py::object EffectiveFrequencyCyl_getVertDeterminant(EffectiveFrequencyCyl& self, py::object val)
{
   return UFUNC<dcomplex>([&](dcomplex x){return self.getVertDeterminant(x);}, val);
}

dcomplex EffectiveFrequencyCyl_getLambda0(const EffectiveFrequencyCyl& self) {
    return 2e3*M_PI / self.k0;
}

void EffectiveFrequencyCyl_setLambda0(EffectiveFrequencyCyl& self, dcomplex lambda0) {
    self.k0 = 2e3*M_PI / lambda0;
}

py::object EffectiveIndex2D_getMirrors(const EffectiveIndex2D& self) {
    if (!self.mirrors) return py::object();
    return py::make_tuple(self.mirrors->first, self.mirrors->second);
}

void EffectiveIndex2D_setMirrors(EffectiveIndex2D& self, py::object value) {
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
                self.mirrors.reset(std::make_pair<double,double>(double(py::extract<double>(value[0])), double(py::extract<double>(value[1]))));
            } catch (py::error_already_set) {
                throw ValueError("None, float, or tuple of two floats required");
            }
        }
    }
}

static size_t EffectiveIndex2D_findMode(EffectiveIndex2D& self, py::object neff, py::object symmetry) {
    return self.findMode(py::extract<dcomplex>(neff), parseSymmetry(symmetry));
}

std::vector<size_t> EffectiveIndex2D_findModes(EffectiveIndex2D& self, dcomplex neff1, dcomplex neff2, py::object symmetry, size_t resteps, size_t imsteps, dcomplex eps) {
    return self.findModes(neff1, neff2, parseSymmetry(symmetry), resteps, imsteps, eps);
}

std::string EffectiveIndex2D_Mode_str(const EffectiveIndex2D::Mode& self) {
    std::string sym;
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: sym = "positive"; break;
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: sym = "negative"; break;
        default: sym = "none";
    }
    return format("<neff: {:.3f}{:+.3g}j, symmetry: {}, power: {:.2g}mW>", real(self.neff), imag(self.neff), sym, self.power);
}
std::string EffectiveIndex2D_Mode_repr(const EffectiveIndex2D::Mode& self) {
    std::string sym;
    switch (self.symmetry) {
        case EffectiveIndex2D::SYMMETRY_POSITIVE: sym = "'positive'"; break;
        case EffectiveIndex2D::SYMMETRY_NEGATIVE: sym = "'negative'"; break;
        default: sym = "None";
    }
    return format("EffectiveIndex2D.Mode(neff={0}, symmetry={1}, power={2})", str(self.neff), sym, self.power);
}


static size_t EffectiveFrequencyCyl_findMode(EffectiveFrequencyCyl& self, py::object lam, int m) {
    return self.findMode(py::extract<dcomplex>(lam), m);
}

double EffectiveFrequencyCyl_Mode_ModalLoss(const EffectiveFrequencyCyl::Mode& mode) {
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
            self.setMesh(plask::make_shared<RectilinearMesh2DFrom1DGenerator>(meshg));
        } catch (py::error_already_set) {
            PyErr_Clear();
            plask::python::detail::ExportedSolverDefaultDefs<SolverT>::Solver_setMesh(self, omesh);
        }
    }
}

py::object EffectiveFrequencyCyl_getStripeR(const EffectiveFrequencyCyl& self) {
    double r = self.getStripeR();
    if (std::isnan(r)) return py::object();
    return py::object(r);
}

void EffectiveFrequencyCyl_setStripeR(EffectiveFrequencyCyl& self, py::object r) {
    if (r == py::object()) self.useAllStripes();
    else self.setStripeR(py::extract<double>(r));
}

std::string EffectiveFrequencyCyl_Mode_str(const EffectiveFrequencyCyl::Mode& self) {
    return format("<m: {:d}, lam: ({:.3f}{:+.3g}j)nm, power: {:.2g}mW>", self.m, real(self.lam), imag(self.lam), self.power);
}
std::string EffectiveFrequencyCyl_Mode_repr(const EffectiveFrequencyCyl::Mode& self) {
    return format("EffectiveFrequencyCyl.Mode(m={0}, lam={1}, power={2})", self.m, str(self.lam), self.power);
}

template <typename Solver>
static double Mode_total_absorption(typename Solver::Mode& self) {
    return self.solver->getTotalAbsorption(self);
}

static double Mode_gain_integral(EffectiveFrequencyCyl::Mode& self) {
    return self.solver->getGainIntegral(self);
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

    {CLASS(EffectiveIndex2D, "EffectiveIndex2D",
        "Calculate optical modes and optical field distribution using the effective index\n"
        "method in two-dimensional Cartesian space.\n")
        solver.add_property("mesh", &EffectiveIndex2D::getMesh, &Optical_setMesh<EffectiveIndex2D>, "Mesh provided to the solver.");
        solver.add_property("polarization", &EffectiveIndex2D_getPolarization, &EffectiveIndex2D_setPolarization, "Polarization of the searched modes.");
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
               "Args:\n"
               SEARCH_ARGS_DOC"\n"
               "Returns:\n"
               "    list of floats: List of the found effective indices in the vertical\n"
               "    direction.\n",
               arg("start")=0., arg("end")=0., arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        solver.def("find_mode", &EffectiveIndex2D_findMode,
                   "Compute the mode near the specified effective index.\n\n"
                   "Args:\n"
                   "    neff (complex): Starting point of the root search.\n"
                   "    symmetry ('+' or '-'): Symmetry of the mode to search.\n\n"
                   "Returns:\n"
                   "    integer: Index in the :attr:`modes` list of the found mode.\n",
                   (arg("neff"), arg("symmetry")=py::object()));
        solver.def("find_modes", &EffectiveIndex2D_findModes,
                   "Find the modes within the specified range using global method.\n\n"
                   "Args:\n"
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
        METHOD(clear_modes, clearModes,
               "Clear all computed modes.\n");
        solver.def("get_total_absorption", (double (EffectiveIndex2D::*)(size_t))&EffectiveIndex2D::getTotalAbsorption,
               "Get total energy absorbed by from a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total absorbed energy [mW].\n",
               py::arg("num")=0);
        RW_PROPERTY(vat, getStripeX, setStripeX, "Horizontal position of the main stripe (with dominant mode).");
        RW_FIELD(vneff, "Effective index in the vertical direction.");
        solver.add_property("mirrors", EffectiveIndex2D_getMirrors, EffectiveIndex2D_setMirrors,
                    "Mirror reflectivities. If None then they are automatically estimated from the"
                    "Fresnel equations.\n");
        solver.def("get_vert_determinant", EffectiveIndex2D_getVertDeterminant,
                   "Get vertical modal determinant for debugging purposes.\n\n"
                   "Args:\n"
                   "    neff (complex of numeric array of complex): Vertical effective index value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the vertical effective index\n"
                   "    *neff* or an array matching its size.",
                   "neff");
        solver.def("get_determinant", &EffectiveIndex2D_getDeterminant,
                   "Get modal determinant.\n\n"
                   "Args:\n"
                   "    neff (complex of numeric array of complex): effective index value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the effective index *neff* or\n"
                   "    an array matching its size.",
                   (py::arg("neff") , py::arg("polarization")=py::object()));
        RW_PROPERTY(wavelength, getWavelength, setWavelength, "Current wavelength.");
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
                 "   ~optical.effective.EffectiveIndex2D.Mode.power\n"
                 "   ~optical.effective.EffectiveIndex2D.Mode.total_absorption\n"
                 ":rtype: Mode\n");

        py::scope scope = solver;

        register_vector_of<EffectiveIndex2D::Mode>("Modes");

        py::class_<EffectiveIndex2D::Mode>("Mode", "Detailed information about the mode.", py::no_init)
            .def_readonly("neff", &EffectiveIndex2D::Mode::neff, "Mode effective index.")
            .add_property("symmetry", &EffectiveIndex2D_getSymmetry, "Mode symmetry ('positive', 'negative', or None).")
            .def_readwrite("power", &EffectiveIndex2D::Mode::power, "Total power emitted into the mode [mW].")
            .add_property("loss", &EffectiveIndex2D::Mode::loss, "Mode losses [1/cm].")
            .add_property("total_absorption", &Mode_total_absorption<EffectiveFrequencyCyl>,
                          "Cumulated absorption for the mode [mW].\n\n"
                          "This property combines gain in active region and absorption in the whole\n"
                          "structure.")
            .def("__str__", &EffectiveIndex2D_Mode_str)
            .def("__repr__", &EffectiveIndex2D_Mode_repr)
        ;

        py_enum<EffectiveIndex2D::Emission>()
            .value("FRONT", EffectiveIndex2D::FRONT)
            .value("BACK", EffectiveIndex2D::BACK)
        ;
    }

    {CLASS(EffectiveFrequencyCyl, "EffectiveFrequencyCyl",
        "Calculate optical modes and optical field distribution using the effective\n"
        "frequency method in two-dimensional cylindrical space.\n")
        solver.add_property("mesh", &EffectiveFrequencyCyl::getMesh, &Optical_setMesh<EffectiveFrequencyCyl>, "Mesh provided to the solver.");
        RW_FIELD(k0, "Reference normalized frequency.");
        RW_FIELD(vlam, "'Vertical wavelength' used as a helper for searching vertical modes.");
        solver.add_property("lam0", &EffectiveFrequencyCyl_getLambda0, &EffectiveFrequencyCyl_setLambda0, "Reference wavelength.");
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
//         RW_PROPERTY(asymptotic, getAsymptotic, setAsymptotic,
//                     "Flag indicating whether the solver uses asymptotic exponential field\n"
//                     "in the outermost layer.")
        RW_PROPERTY(emission, getEmission, setEmission, "Emission direction.");
        METHOD(set_simple_mesh, setSimpleMesh, "Set simple mesh based on the geometry objects bounding boxes.");
        // METHOD(set_horizontal_mesh, setHorizontalMesh, "Set custom mesh in horizontal direction, vertical one is based on the geometry objects bounding boxes", "points");
        solver.def("find_mode", &EffectiveFrequencyCyl_findMode,
                   "Compute the mode near the specified wavelength.\n\n"
                   "Args:\n"
                   "    lam (complex): Initial wavelength to for root finging algorithm.\n"
                   "    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n"
                   "Returns:\n"
                   "    integer: Index in the :attr:`modes` list of the found mode.\n",
                   (arg("lam"), arg("m")=0));
        METHOD(find_modes, findModes,
               "Find the modes within the specified range using global method.\n\n"
               "Args:\n"
               "    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n"
               SEARCH_ARGS_DOC"\n"
               "Returns:\n"
               "    list of integers: List of the indices in the :attr:`modes` list of the found\n"
               "    modes.\n",
               arg("start")=0., arg("end")=0., arg("m")=0, arg("resteps")=256, arg("imsteps")=64, arg("eps")=dcomplex(1e-6, 1e-9));
        solver.def("get_vert_determinant", &EffectiveFrequencyCyl_getVertDeterminant,
                   "Get vertical modal determinant for debugging purposes.\n\n"
                   "Args:\n"
                   "    vlam (complex of numeric array of complex): Vertical wavelength value\n"
                   "    to compute the determinant at.\n\n"
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the vertical wavelength *vlam* or\n"
                   "    an array matching its size.\n",
                   py::arg("vlam"));
        solver.def("get_determinant", &EffectiveFrequencyCyl_getDeterminant,
                   "Get modal determinant.\n\n"
                   "Args:\n"
                   "    lam (complex of numeric array of complex): wavelength to compute the\n"
                   "                                               determinant at.\n"
                   "    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n\n",
                   "Returns:\n"
                   "    complex or list of complex: Determinant at the effective index *neff* or\n"
                   "    an array matching its size.\n",
                   (py::arg("lam"), py::arg("m")=0));
        solver.def("set_mode", (size_t (EffectiveFrequencyCyl::*)(dcomplex,int))&EffectiveFrequencyCyl::setMode, (py::arg("lam"), py::arg("m")=0));
        solver.def("set_mode", (size_t (EffectiveFrequencyCyl::*)(double,double,int))&EffectiveFrequencyCyl::setMode,
                   "Set the current mode the specified wavelength.\n\n"
                   "Args:\n"
                   "    lam (float of complex): Mode wavelength.\n"
                   "    loss (float): Mode losses. Allowed only if *lam* is a float.\n"
                   "    m (int): Angular mode number (O for LP0x, 1 for LP1x, etc.).\n",
                   (py::arg("lam"), "loss", py::arg("m")=0));
        METHOD(clear_modes, clearModes,
               "Clear all computed modes.\n");
        solver.def("get_total_absorption", (double (EffectiveFrequencyCyl::*)(size_t))&EffectiveFrequencyCyl::getTotalAbsorption,
               "Get total energy absorbed from a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total absorbed energy [mW].\n",
               py::arg("num")=0);
        solver.def("get_gain_integral", (double (EffectiveFrequencyCyl::*)(size_t))&EffectiveFrequencyCyl::getGainIntegral,
               "Get total energy generated in the gain region to a mode in unit time.\n\n"
               "Args:\n"
               "    num (int): number of the mode.\n\n"
               "Returns:\n"
               "    Total generated energy [mW].\n",
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
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.power\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.total_absorption\n"
                 "   ~optical.effective.EffectiveFrequencyCyl.Mode.gain_integral\n"
                 ":rtype: Mode\n");
        solver.add_property("vat", &EffectiveFrequencyCyl_getStripeR, &EffectiveFrequencyCyl_setStripeR,
                            "Radial position of at which the vertical part of the field is calculated.\n\n"
                            "Should be a float number or ``None`` to compute effective frequencies for all\n"
                            "the stripes.\n");

        py::scope scope = solver;

        register_vector_of<EffectiveFrequencyCyl::Mode>("Modes");

        py::class_<EffectiveFrequencyCyl::Mode>("Mode", "Detailed information about the mode.", py::no_init)
            .def_readonly("m", &EffectiveFrequencyCyl::Mode::m, "LP_mn mode parameter describing angular dependence.")
            .def_readonly("lam", &EffectiveFrequencyCyl::Mode::lam, "Alias for :attr:`~optical.effective.EffectiveFrequencyCyl.Mode.wavelength`.")
            .def_readonly("wavelength", &EffectiveFrequencyCyl::Mode::lam, "Mode wavelength [nm].")
            .def_readwrite("power", &EffectiveFrequencyCyl::Mode::power, "Total power emitted into the mode.")
            .add_property("loss", &EffectiveFrequencyCyl::Mode::loss, "Mode losses [1/cm].")
            .add_property("total_absorption", &Mode_total_absorption<EffectiveFrequencyCyl>,
                          "Cumulated absorption for the mode [mW].\n\n"
                          "This property combines gain in active region and absorption in the whole\n"
                          "structure.")
            .add_property("gain_integral", &Mode_gain_integral, "Total gain for the mode [mW].")
            .def("__str__", &EffectiveFrequencyCyl_Mode_str)
            .def("__repr__", &EffectiveFrequencyCyl_Mode_repr)
        ;

        py_enum<EffectiveFrequencyCyl::Emission>()
            .value("TOP", EffectiveFrequencyCyl::TOP)
            .value("BOTTOM", EffectiveFrequencyCyl::BOTTOM)
        ;
    }

    py::class_<RootDigger::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("method", &RootDigger::Params::method, "Root finding method ('muller', 'broyden', or 'brent')")
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, "Maximum number of iterations.")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration (Broyden method only).")
        .def_readwrite("alpha", &RootDigger::Params::maxstep, "Parameter ensuring sufficient decrease of determinant in each step\n(Broyden method only).")
        .def_readwrite("lambd", &RootDigger::Params::maxstep, "Minimum decrease ratio of one step (Broyden method only).")
        .def_readwrite("initial_range", &RootDigger::Params::initial_dist, "Initial range size (Muller and Brent methods only).")
    ;

    py_enum<RootDigger::Method>()
        .value("MULLER", RootDigger::ROOT_MULLER)
        .value("BROYDEN", RootDigger::ROOT_BROYDEN)
        .value("BRENT", RootDigger::ROOT_BRENT)
    ;
}
