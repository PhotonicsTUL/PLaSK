#include "fourier2d-python.h"
#include "slab-python.h"


namespace plask { namespace solvers { namespace slab { namespace python {


static py::object FourierSolver2D_getMirrors(const FourierSolver2D& self) {
    if (!self.mirrors) return py::object();
    return py::make_tuple(self.mirrors->first, self.mirrors->second);
}


static void FourierSolver2D_setMirrors(FourierSolver2D& self, py::object value) {
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


static py::object FourierSolver2D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
        WHAT_NEFF,
        WHAT_KTRAN
    };
    What what = WHAT_NOTHING;
    py::object array;

    FourierSolver2D::ParamGuard guard(self);

    AxisNames* axes = getCurrentAxes();
    boost::optional<dcomplex> lambda, neff, ktran;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                lambda.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0")
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                lambda.reset(2e3*M_PI / dcomplex(py::extract<dcomplex>(kwargs[*i])));
        else if (*i == "neff")
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_NEFF; array = kwargs[*i];
            } else
                neff.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran())
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_KTRAN; array = kwargs[*i];
            } else
                ktran.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "dispersive")
            throw TypeError("Dispersive argument has been removed: set solver.lam0 attribute");
        else
            throw TypeError("get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    if (lambda) self->setWavelength(*lambda);
    if (neff) self->setKlong(*neff * self->getK0());
    if (ktran) self->setKtran(*ktran);

    switch (what) {
        case WHAT_NOTHING:
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setWavelength(x); return self->getDeterminant(); },
                array
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setK0(x); return self->getDeterminant(); },
                array
            );
        case WHAT_NEFF:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setKlong(x * self->getK0()); return self->getDeterminant(); },
                array
            );
        case WHAT_KTRAN:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setKtran(x); return self->getDeterminant(); },
                array
            );
    }
    return py::object();
}

static size_t FourierSolver2D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("find_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError("find_mode() takes exactly one keyword argument ({0} given)", py::len(kwargs));
    std::string key = py::extract<std::string>(kwargs.keys()[0]);
    dcomplex value = py::extract<dcomplex>(kwargs[key]);
    AxisNames* axes = getCurrentAxes();
    FourierSolver2D::What what;

    if (key == "lam" || key == "wavelength")
        what = FourierSolver2D::WHAT_WAVELENGTH;
    else if (key == "k0")
        what = FourierSolver2D::WHAT_K0;
    else if (key == "neff")
        what = FourierSolver2D::WHAT_NEFF;
    else if (key == "ktran" || key == "kt" || key == "k"+axes->getNameForTran())
        what = FourierSolver2D::WHAT_KTRAN;
    else
        throw TypeError("find_mode() got unexpected keyword argument '{0}'", key);

    return self->findMode(what, value);
}


static dcomplex FourierSolver2D_Mode_Neff(const FourierSolver2D::Mode& mode) {
    return mode.beta / mode.k0;
}

static py::object FourierSolver2D_Mode__getattr__(const FourierSolver2D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.beta);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError("'Mode' object has no attribute '{0}'", name);
    return py::object();
}


static std::string FourierSolver2D_Mode_str(const FourierSolver2D::Mode& self) {
    AxisNames* axes = getCurrentAxes();
    std::string pol;
    switch (self.polarization) {
        case Expansion::E_TRAN: pol = "E" + axes->getNameForTran(); break;
        case Expansion::E_LONG: pol = "E" + axes->getNameForLong(); break;
        default: pol = "none";
    }
    std::string sym;
    switch (self.symmetry) {
        case Expansion::E_TRAN: sym = "E" + axes->getNameForTran(); break;
        case Expansion::E_LONG: sym = "E" + axes->getNameForLong(); break;
        default: sym = "none";
    }
    dcomplex neff = self.beta / self.k0;
    return format("<lam: {:.2f}nm, neff: {:.3f}{:+.3g}, ktran: {}/um, polarization: {}, symmetry: {}, power: {:.2g} mW>",
                  real(2e3*M_PI / self.k0),
                  real(neff),imag(neff),
                  (imag(self.ktran) == 0.)? format("{:.3g}",real(self.ktran)) : format("{:.3g}{:+.3g}",real(self.ktran),imag(self.ktran)),
                  pol,
                  sym,
                  self.power
                 );
}
static std::string FourierSolver2D_Mode_repr(const FourierSolver2D::Mode& self) {
    AxisNames* axes = getCurrentAxes();
    std::string pol;
    switch (self.polarization) {
        case Expansion::E_TRAN: pol = "'E" + axes->getNameForTran() + "'"; break;
        case Expansion::E_LONG: pol = "'E" + axes->getNameForLong() + "'"; break;
        default: pol = "None";
    }
    std::string sym;
    switch (self.symmetry) {
        case Expansion::E_TRAN: sym = "'E" + axes->getNameForTran() + "'"; break;
        case Expansion::E_LONG: sym = "'E" + axes->getNameForLong() + "'"; break;
        default: sym = "None";
    }
    return format("Fourier2D.Mode(lam={0}, neff={1}, ktran={2}, polarization={3}, symmetry={4}, power={5})",
                  str(2e3*M_PI/self.k0), str(self.beta/self.k0), str(self.ktran), pol, sym, self.power);
}

static py::object FourierSolver2D_reflectedAmplitudes(FourierSolver2D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    FourierSolver2D::ParamGuard guard(&self);
    self.setWavelength(lam);
    auto data = self.getReflectedAmplitudes(polarization, incidence);
    return arrayFromVec2D<NPY_DOUBLE>(data, self.separated());
}

static py::object FourierSolver2D_transmittedAmplitudes(FourierSolver2D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    FourierSolver2D::ParamGuard guard(&self);
    self.setWavelength(lam);
    auto data = self.getTransmittedAmplitudes(polarization, incidence);
    return arrayFromVec2D<NPY_DOUBLE>(data, self.separated());
}

static py::object FourierSolver2D_getFieldVectorE(FourierSolver2D& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), self.separated(), 2);
}

static py::object FourierSolver2D_getFieldVectorH(FourierSolver2D& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), self.separated(), 2);
}

static py::object FourierSolver2D_getReflectedFieldVectorE(FourierSolver2D::Reflected& self, double z) {
    return arrayFromVec2D<NPY_CDOUBLE>(self.parent->getReflectedFieldVectorE(self.polarization, self.side, z), self.parent->separated(), 2);
}

static py::object FourierSolver2D_getReflectedFieldVectorH(FourierSolver2D::Reflected& self, double z) {
    return arrayFromVec2D<NPY_CDOUBLE>(self.parent->getReflectedFieldVectorH(self.polarization, self.side, z), self.parent->separated(), 2);
}



void export_FourierSolver2D()
{
    plask_import_array();

    CLASS(FourierSolver2D, "Fourier2D",
        "Optical Solver using Fourier expansion in 2D.\n\n"
        "It calculates optical modes and optical field distribution using Fourier slab method\n"
        "and reflection transfer in two-dimensional Cartesian space.")
    export_base(solver);
    solver.add_property("material_mesh", &__Class__::getMesh, 
                "Regular mesh with points in which material is sampled.");
    PROVIDER(outNeff, "Effective index of the last computed mode.");
    solver.def("find_mode", py::raw_function(FourierSolver2D_findMode),
                "Compute the mode near the specified effective index.\n\n"
                "Only one of the following arguments can be given through a keyword.\n"
                "It is the starting point for search of the specified parameter.\n\n"
                "Args:\n"
                "    lam (complex): Wavelength.\n"
                "    k0 (complex): Normalized frequency.\n"
                "    neff (complex): Longitudinal effective index.\n"
                "    ktran (complex): Transverse wavevector.\n");
    RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size.");
    RW_PROPERTY(symmetry, getSymmetry, setSymmetry, "Mode symmetry.");
    RW_PROPERTY(polarization, getPolarization, setPolarization, "Mode polarization.");
    solver.add_property("wavelength", &__Class__::getWavelength, &Solver_setWavelength<__Class__>, 
                "Wavelength of the light [nm].\n\n"
                "Use this property only if you are looking for anything else than\n"
                "the wavelength, e.g. the effective index of lateral wavevector.\n");
    solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>, 
                "Normalized frequency of the light [1/µm].\n\n"
                "Use this property only if you are looking for anything else than\n"
                "the wavelength,e.g. the effective index of lateral wavevector.\n");
    RW_PROPERTY(klong, getKlong, setKlong,
                "Longitudinal propagation constant of the light [1/µm].\n\n"
                "Use this property only if you are looking for anything else than\n"
                "the longitudinal component of the propagation vector and the effective index.\n");
    RW_PROPERTY(ktran, getKtran, setKtran,
                "Transverse propagation constant of the light [1/µm].\n\n"
                "Use this property only  if you are looking for anything else than\n"
                "the transverse component of the propagation vector.\n");
    RW_FIELD(refine, "Number of refinement points for refractive index averaging.");
    RW_FIELD(oversampling, "Factor by which the number of coefficients is increased for FFT.");
    solver.add_property("dct", &__Class__::getDCT, &__Class__::setDCT, 
                "Type of discrete cosine transform for symmetric expansion.");
    solver.def("get_determinant", py::raw_function(FourierSolver2D_getDeterminant),
                "Compute discontinuity matrix determinant.\n\n"
                "Arguments can be given through keywords only.\n\n"
                "Args:\n"
                "    lam (complex): Wavelength.\n"
                "    k0 (complex): Normalized frequency.\n"
                "    neff (complex): Longitudinal effective index.\n"
                "    ktran (complex): Transverse wavevector.\n");
    solver.def("compute_reflectivity", &FourierSolver_computeReflectivity<FourierSolver2D>,
                "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
                "Args:\n"
                "    lam (float or array of floats): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_transmittivity", &FourierSolver_computeTransmittivity<FourierSolver2D>,
                "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
                "Args:\n"
                "    lam (float or array of floats): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                "        of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_reflected_orders", &FourierSolver2D_reflectedAmplitudes,
                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_transmitted_orders", &FourierSolver2D_transmittedAmplitudes,
                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.add_property("mirrors", FourierSolver2D_getMirrors, FourierSolver2D_setMirrors,
                "Mirror reflectivities. If None then they are automatically estimated from the\n"
                "Fresnel equations.");
    solver.add_property("pml", py::make_function(&Solver_getPML<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setPML<FourierSolver2D>,
                        "Side Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC
                        );
    RO_FIELD(modes, "Computed modes.");
    solver.def("reflected", FourierSolver_getReflected<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>(),
               "Access to the reflected field.\n\n"
               "Args:\n"
               "    lam (float): Incident light wavelength.\n"
               "    polarization: Specification of the incident light polarization.\n"
               "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               "        of the non-vanishing electric field component.\n"
               "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               "        present.\n\n"
               ":rtype: Fourier2D.Reflected\n"
               , (py::arg("lam"), "polarization", "side")
              );
    solver.def("get_electric_coefficients", FourierSolver2D_getFieldVectorE, (py::arg("num"), "level"),
               "Get Fourier expansion coefficients for the electric field.\n\n"
               "This is a low-level function returning $E_l$ and/or $E_t$ Fourier expansion\n"
               "coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical level at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
    solver.def("get_magnetic_coefficients", FourierSolver2D_getFieldVectorH, (py::arg("num"), "level"),
               "Get Fourier expansion coefficients for the magnetic field.\n\n"
               "This is a low-level function returning $H_l$ and/or $H_t$ Fourier expansion\n"
               "coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical level at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
    py::scope scope = solver;

    register_vector_of<FourierSolver2D::Mode>("Modes");
    py::class_<FourierSolver2D::Mode>("Mode", "Detailed information about the mode.", py::no_init)
        .def_readonly("symmetry", &FourierSolver2D::Mode::symmetry, "Mode horizontal symmetry.")
        .def_readonly("polarization", &FourierSolver2D::Mode::polarization, "Mode polarization.")
        .add_property("lam", &getModeWavelength<FourierSolver2D::Mode>, "Mode wavelength [nm].")
        .add_property("wavelength", &getModeWavelength<FourierSolver2D::Mode>, "Mode wavelength [nm].")
        .def_readonly("k0", &FourierSolver2D::Mode::k0, "Mode normalized frequency [1/µm].")
        .def_readonly("beta", &FourierSolver2D::Mode::beta, "Mode longitudinal wavevector [1/µm].")
        .add_property("neff", &FourierSolver2D_Mode_Neff, "Mode longitudinal effective index [-].")
        .def_readonly("ktran", &FourierSolver2D::Mode::ktran, "Mode transverse wavevector [1/µm].")
        .def_readwrite("power", &FourierSolver2D::Mode::power, "Total power emitted into the mode.")
        .def("__str__", &FourierSolver2D_Mode_str)
        .def("__repr__", &FourierSolver2D_Mode_repr)
        .def("__getattr__", &FourierSolver2D_Mode__getattr__)
    ;

    py::class_<FourierSolver2D::Reflected, shared_ptr<FourierSolver2D::Reflected>, boost::noncopyable>("Reflected",
        "Reflected mode proxy.\n\n"
        "This class contains providers for the optical field for a reflected field"
        "under the normal incidence.\n"
        , py::no_init)
        .def_readonly("outElectricField", reinterpret_cast<ProviderFor<LightE,Geometry2DCartesian> FourierSolver2D::Reflected::*>
                                            (&FourierSolver2D::Reflected::outElectricField),
            format(docstring_attr_provider<LightE>(), "LightE", "2D", "electric field", "V/m", "", "", "", "outElectricField").c_str()
        )
        .def_readonly("outMagneticField", reinterpret_cast<ProviderFor<LightH,Geometry2DCartesian> FourierSolver2D::Reflected::*>
                                            (&FourierSolver2D::Reflected::outMagneticField),
            format(docstring_attr_provider<LightH>(), "LightH", "2D", "magnetic field", "A/m", "", "", "", "outMagneticField").c_str()
        )
        .def_readonly("outLightMagnitude", reinterpret_cast<ProviderFor<LightMagnitude,Geometry2DCartesian> FourierSolver2D::Reflected::*>
                                            (&FourierSolver2D::Reflected::outLightMagnitude),
            format(docstring_attr_provider<LightMagnitude>(), "LightMagnitude", "2D", "light intensity", "W/m²", "", "", "", "outLightMagnitude").c_str()
        )
        .def("get_electric_coefficients", FourierSolver2D_getReflectedFieldVectorE, py::arg("level"),
             "Get Fourier expansion coefficients for the electric field.\n\n"
             "This is a low-level function returning $E_l$ and/or $E_t$ Fourier expansion\n"
             "coefficients. Please refer to the detailed solver description for their\n"
             "interpretation.\n\n"
             "Args:\n"
             "    level (float): Vertical level at which the coefficients are computed.\n\n"
             ":rtype: numpy.ndarray\n"
            )
        .def("get_magnetic_coefficients", FourierSolver2D_getReflectedFieldVectorH, py::arg("level"),
             "Get Fourier expansion coefficients for the magnegtic field.\n\n"
             "This is a low-level function returning $H_l$ and/or $H_t$ Fourier expansion\n"
             "coefficients. Please refer to the detailed solver description for their\n"
             "interpretation.\n\n"
             "Args:\n"
             "    level (float): Vertical level at which the coefficients are computed.\n\n"
             ":rtype: numpy.ndarray\n"
            )
    ;
}

}}}} // namespace plask::solvers::slab::python