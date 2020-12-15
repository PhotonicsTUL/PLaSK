#define PY_ARRAY_UNIQUE_SYMBOL PLASK_OPTICAL_SLAB_ARRAY_API
#define NO_IMPORT_ARRAY

#include "plask/python_numpy.hpp"

#include "besselcyl-python.hpp"
#include "slab-python.hpp"

namespace plask { namespace optical { namespace slab { namespace python {

template <> inline const char* solver_compute_reflectivity_name<BesselSolverCyl>() {
    return "BesselCyl.compute_reflectivity";
}

template <> inline const char* solver_compute_transmittivity_name<BesselSolverCyl>() {
    return "BesselCyl.compute_transmittivity";
}


template <>
py::object Eigenmodes<BesselSolverCyl>::array(const dcomplex* data, size_t N) const {
    const int dim = 2, strid = 2;
    npy_intp dims[] = { npy_intp(N / strid), npy_intp(strid) };
    npy_intp strides[] = { strid * sizeof(dcomplex), sizeof(dcomplex) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, NPY_CDOUBLE, strides, (void*)data, 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array");
    return py::object(py::handle<>(arr));
}

std::string BesselSolverCyl_Mode_str(const BesselSolverCyl::Mode& self) {
    return format(u8"<m: {:d}, lam: {}nm, power: {:.2g}mW>", self.m, str(2e3*PI/self.k0, u8"({:.3f}{:+.3g}j)"), self.power);
}
std::string BesselSolverCyl_Mode_repr(const BesselSolverCyl::Mode& self) {
    return format(u8"BesselCyl.Mode(m={:d}, lam={}, power={:g})", self.m, str(2e3*PI/self.k0), self.power);
}

py::object BesselSolverCyl_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
    BesselSolverCyl* self = py::extract<BesselSolverCyl*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
    };
    What what = WHAT_NOTHING;
    py::object array;
    int m = self->getM();

    plask::optional<dcomplex> k0;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength") {
            if (what == WHAT_K0 || k0)
                throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                k0.reset(2e3*PI / py::extract<dcomplex>(kwargs[*i])());
        } else if (*i == "k0") {
            if (what == WHAT_WAVELENGTH || k0)
                throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                k0.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "m") {
            m = py::extract<int>(kwargs[*i]);
        } else if (*i == "dispersive") {
            throw TypeError(u8"Dispersive argument has been removed: set solver.lam0 attribute");
        } else
            throw TypeError(u8"get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();
    auto* expansion = self->expansion.get();

    if (k0) expansion->setK0(*k0);
    expansion->setM(m);

    switch (what) {
        case WHAT_NOTHING:
            if (!k0) expansion->setK0(self->getK0());
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion->setK0(2e3*PI / x); return self->getDeterminant(); },
                array,
                "BesselCyl.get_determinant",
                "lam"
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion->setK0(x); return self->getDeterminant(); },
                array,
                "BesselCyl.get_determinant",
                "k0"
            );
    }
    return py::object();
}

static size_t BesselSolverCyl_findMode(BesselSolverCyl& self, dcomplex start, const py::object& pym) {
    int m;
    if (pym.is_none()) {
        m = self.getM();
    } else {
        m = py::extract<int>(pym);
    }
    return self.findMode(start, m);
}

static size_t BesselSolverCyl_setMode(BesselSolverCyl* self, dcomplex lam, const py::object& pym) {
    self->Solver::initCalculation();
    self->setExpansionDefaults();

    auto* expansion = self->expansion.get();
    expansion->setK0(2e3*PI / lam);

    if (!pym.is_none()) {
        expansion->setM(py::extract<int>(pym));
    }

    return self->setMode();
}

static py::object BesselSolverCyl_getFieldVectorE(BesselSolverCyl& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), false, 2);
}

static py::object BesselSolverCyl_getFieldVectorH(BesselSolverCyl& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), false, 2);
}

static py::object BesselSolverCyl_getKweights(BesselSolverCyl& self) {
    if (self.getDomain() == BesselSolverCyl::DOMAIN_INFINITE)
        return py::object(self.getKweights());
    return py::object();
}

static void BesselSolverCyl_setKweights(BesselSolverCyl& self, const py::object& value) {
    if (value.is_none()) self.clearKweights();
    else self.setKweights(py::extract<std::vector<double>>(value));
}


void export_BesselSolverCyl()
{
    py_enum<typename BesselSolverCyl::BesselDomain>()
        .value("FINITE", BesselSolverCyl::DOMAIN_FINITE)
        .value("INFINITE", BesselSolverCyl::DOMAIN_INFINITE)
    ;

    py_enum<typename BesselSolverCyl::Rule>()
        .value("INVERSE", BesselSolverCyl::RULE_INVERSE_0)
        .value("INVERSE1", BesselSolverCyl::RULE_INVERSE_1)
        .value("INVERSE2", BesselSolverCyl::RULE_INVERSE_2)
        .value("DIRECT", BesselSolverCyl::RULE_DIRECT)
    ;

    py_enum<typename BesselSolverCyl::InfiniteWavevectors>()
        .value("UNIFORM", BesselSolverCyl::WAVEVECTORS_UNIFORM)
        .value("NONUNIFORM", BesselSolverCyl::WAVEVECTORS_NONUNIFORM)
        .value("NON_UNIFORM", BesselSolverCyl::WAVEVECTORS_NONUNIFORM)
        .value("LAGUERRE", BesselSolverCyl::WAVEVECTORS_LAGUERRE)
        .value("MANUAL", BesselSolverCyl::WAVEVECTORS_MANUAL)
    ;

    CLASS(BesselSolverCyl, "BesselCyl",
        u8"Optical Solver using Bessel expansion in cylindrical coordinates.\n\n"
        u8"It calculates optical modes and optical field distribution using Bessel slab method\n"
        u8"and reflection transfer in two-dimensional cylindrical space.")
        export_base(solver);
//     solver.add_property("material_mesh", &__Class__::getMesh, u8"Regular mesh with points in which material is sampled.");
    PROVIDER(outLoss, "");
    RW_PROPERTY(domain, getDomain, setDomain, u8"Computational domain ('finite' or 'infinite').");
    RW_PROPERTY(rule, getRule, setRule,
        u8"Expansion rule for coefficients matrix. Can be 'direct', 'semi-inverse'\n"
        u8"or 'inverse'. Inverse rule is proven to provide the best convergence and\n"
        u8"should be used in almost every case.\n");
    RW_PROPERTY(size, getSize, setSize, u8"Orthogonal expansion size.");
    RW_PROPERTY(kmethod, getKmethod, setKmethod,
        u8"Method of selecting wavevectors for numerical Hankel transform in infinite\n"
        u8"domain.");
    RW_PROPERTY(klist, getKlist, setKlist,
            u8"A list of wavevectors ranges. If no weights are given, the actual wavevectors\n"
            u8"used in the computations are the averages of each two adjacent values specified\n"
            u8"here and the integration weights are the sizes of each interval.\n");
    solver.add_property("kweights", &BesselSolverCyl_getKweights, &BesselSolverCyl_setKweights,
            u8"An optional list of relative wavevector weights. The numbers should be relative\n"
            u8"to the inverse of the structure width.");
    RW_PROPERTY(kmax, getKmax, setKmax,
                u8"Maximum wavevector used in the infinite domain relative to the wavelength.\n");
    RW_PROPERTY(kscale, getKscale, setKscale,
                u8"Scale factor for wavevectors used in the infinite domain.\n");
    solver.add_property("lam", &__Class__::getLam, &Solver_setLam<__Class__>,
                u8"Wavelength of the light [nm].\n");
    solver.add_property("wavelength", &__Class__::getLam, &Solver_setLam<__Class__>,
                u8"Alias for :attr:`lam`");
    solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>,
                u8"Normalized frequency of the light [1/µm].\n");
    solver.add_property("m", &__Class__::getM, &__Class__::setM, "Angular dependence parameter.");
    solver.def("find_mode", &BesselSolverCyl_findMode,
           u8"Compute the mode near the specified effective index.\n\n"
           u8"Only one of the following arguments can be given through a keyword.\n"
           u8"It is the starting point for search of the specified parameter.\n\n"
           u8"Args:\n"
           u8"    lam (complex): Starting wavelength.\n"
           u8"    m (int): HE/EH Mode angular number. If ``None``, use :attr:`m` attribute.\n",
           (arg("lam"), arg("m")=py::object())
          );
    solver.def("set_mode", &BesselSolverCyl_setMode,
                u8"Set the mode for specified parameters.\n\n"
                u8"This method should be used if you have found a mode manually and want to insert\n"
                u8"it into the solver in order to determine the fields. Calling this will raise an\n"
                u8"exception if the determinant for the specified parameters is too large.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    m (int): HE/EH Mode angular number.\n"
              );
    RW_FIELD(emission, "Direction of the useful light emission.\n\n"
                       u8"Necessary for the over-threshold model to correctly compute the output power.\n"
                       u8"Currently the fields are normalized only if this parameter is set to\n"
                       u8"``top`` or ``bottom``. Otherwise, it is ``undefined`` (default) and the fields\n"
                       u8"are not normalized.");
    solver.def("get_determinant", py::raw_function(BesselSolverCyl_getDeterminant),
               u8"Compute discontinuity matrix determinant.\n\n"
               u8"Arguments can be given through keywords only.\n\n"
               u8"Args:\n"
               u8"    lam (complex): Wavelength.\n"
               u8"    k0 (complex): Normalized frequency.\n"
               u8"    m (int): HE/EH Mode angular number.\n"
              );
    solver.def("compute_reflectivity", &Solver_computeReflectivity_index<BesselSolverCyl>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_reflectivity", &Solver_computeReflectivity_array<BesselSolverCyl>,
               (py::arg("lam"), "side", "coeffs"),
               u8"Compute reflection coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    index: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_index<BesselSolverCyl>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_array<BesselSolverCyl>,
               (py::arg("lam"), "side", "coeffs"),
               u8"Compute transmission coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    index: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("scattering", Scattering<BesselSolverCyl>::from_index, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "idx"));
    solver.def("scattering", Scattering<BesselSolverCyl>::from_array, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "coeffs"),
               u8"Access to the reflected field.\n\n"
               u8"Args:\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               u8"        of the non-vanishing electric field component.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n\n"
               u8":rtype: Fourier2D.Scattering\n"
              );
    solver.def("get_raw_E", BesselSolverCyl_getFieldVectorE, (py::arg("num"), "level"),
               u8"Get Bessel expansion coefficients for the electric field.\n\n"
               u8"This is a low-level function returning :math:`E_s` and :math:`E_p` Bessel\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical lever at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("get_raw_H", BesselSolverCyl_getFieldVectorH, (py::arg("num"), "level"),
               u8"Get Bessel expansion coefficients for the magnetic field.\n\n"
               u8"This is a low-level function returning :math:`H_s` and :math:`H_p` Bessel\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical lever at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.add_property("pml", py::make_function(&Solver_getPML<BesselSolverCyl>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setPML<BesselSolverCyl>,
                        "Side Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC
                       );
    RO_FIELD(modes, "Computed modes.");

    solver.def("layer_eigenmodes", &Eigenmodes<BesselSolverCyl>::fromZ, py::arg("level"),
        u8"Get eigenmodes for a layer at specified level.\n\n"
        u8"This is a low-level function to access diagonalized eigenmodes for a specific\n"
        u8"layer. Please refer to the detailed solver description for the interpretation\n"
        u8"of the returned values.\n\n"
        u8"Args:\n"
        u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
        u8":rtype: :class:`~optical.slab.BesselCyl.Eigenmodes`\n",
        py::with_custodian_and_ward_postcall<0, 1>()
    );

#ifndef NDEBUG
    solver.def("epsV_k", &BesselSolverCyl::epsV_k, py::arg("layer"));
    solver.def("epsTss", &BesselSolverCyl::epsTss, py::arg("layer"));
    solver.def("epsTsp", &BesselSolverCyl::epsTsp, py::arg("layer"));
    solver.def("epsTps", &BesselSolverCyl::epsTps, py::arg("layer"));
    solver.def("epsTpp", &BesselSolverCyl::epsTpp, py::arg("layer"));
    solver.def("muV_k", &BesselSolverCyl::muV_k);
    solver.def("muTss", &BesselSolverCyl::muTss);
    solver.def("muTsp", &BesselSolverCyl::muTsp);
    solver.def("muTps", &BesselSolverCyl::muTps);
    solver.def("muTpp", &BesselSolverCyl::muTpp);
#endif

    py::scope scope = solver;
    (void) scope;   // don't warn about unused variable scope

    register_vector_of<BesselSolverCyl::Mode>("Modes");
    py::class_<BesselSolverCyl::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .add_property("lam", &getModeWavelength<BesselSolverCyl::Mode>, u8"Mode wavelength [nm].")
        .add_property("loss", &getModeLoss<BesselSolverCyl::Mode>, u8"Mode loss [1/cm].")
        .add_property("wavelength", &getModeWavelength<BesselSolverCyl::Mode>, u8"Mode wavelength [nm].")
        .def_readonly("k0", &BesselSolverCyl::Mode::k0, u8"Mode normalized frequency [1/µm].")
        .def_readonly("m", &BesselSolverCyl::Mode::m, u8"Angular mode order.")
        .def_readwrite("power", &BesselSolverCyl::Mode::power, u8"Total power emitted into the mode.")
        .def("__str__", &BesselSolverCyl_Mode_str)
        .def("__repr__", &BesselSolverCyl_Mode_repr)
    ;

    Scattering<BesselSolverCyl>::registerClass("Cyl");
    Eigenmodes<BesselSolverCyl>::registerClass("BesselCyl", "Cyl");
}

}}}} // namespace plask::optical::slab::python
