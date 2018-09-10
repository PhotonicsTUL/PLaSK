#define PY_ARRAY_UNIQUE_SYMBOL PLASK_OPTICAL_SLAB_ARRAY_API
#define NO_IMPORT_ARRAY

#include "fourier2d-python.h"
#include "slab-python.h"


namespace plask { namespace optical { namespace slab { namespace python {

template <>
py::object Eigenmodes<FourierSolver2D>::array(const dcomplex* data, size_t N) const {
    int dim = 2, strid = 2;
    if (solver.separated()) strid = dim = 1;
    npy_intp dims[] = { npy_intp(N / strid), npy_intp(strid) };
    npy_intp strides[] = { npy_intp(strid * sizeof(dcomplex)), npy_intp(sizeof(dcomplex)) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, NPY_CDOUBLE, strides, (void*)data, 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array");
    return py::object(py::handle<>(arr));
}


inline static std::string polarization_str(Expansion::Component val) {
    AxisNames* axes = getCurrentAxes();
    switch (val) {
        case Expansion::E_TRAN: return "E"+axes->getNameForTran();
        case Expansion::E_LONG: return "E"+axes->getNameForLong();
        default: return "none";
    }
}


template <>
py::object Solver_computeReflectivity<FourierSolver2D>(FourierSolver2D* self,
                                                       py::object wavelength,
                                                       Transfer::IncidentDirection side,
                                                       Expansion::Component polarization
                                                      )
{
    if (self->getBeta() == 0. && (!self->expansion.initialized || self->expansion.separated())) {
        if (!self->isInitialized()) {
            if (self->getPolarization() != polarization) {
                self->writelog(LOG_WARNING, u8"Changing polarization to {0} (manually initialize solver to disable)",
                               polarization_str(polarization));
                self->setPolarization(polarization);
            }
            self->Solver::initCalculation();
        } else {
            self->expansion.setLam0(self->getLam0());
            self->expansion.setBeta(self->getBeta());
            self->expansion.setKtran(self->getKtran());
            self->expansion.setSymmetry(self->getSymmetry());
            self->expansion.setPolarization(polarization);
        }
    } else if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->incidentVector(side, polarization, k0);
        self->expansion.setK0(k0);
        return 100. * self->getReflection(incident, side);
    }, wavelength);
}

template <>
py::object Solver_computeTransmittivity<FourierSolver2D>(FourierSolver2D* self,
                                                         py::object wavelength,
                                                         Transfer::IncidentDirection side,
                                                         Expansion::Component polarization
                                                        )
{
    if (self->getBeta() == 0. && (!self->expansion.initialized || self->expansion.separated())) {
        if (!self->isInitialized()) {
            if (self->getPolarization() != polarization) {
                self->writelog(LOG_WARNING, u8"Changing polarization to {0} (manually initialize solver to disable)",
                               polarization_str(polarization));
                self->setPolarization(polarization);
            }
            self->Solver::initCalculation();
        } else {
            self->expansion.setLam0(self->getLam0());
            self->expansion.setBeta(self->getBeta());
            self->expansion.setKtran(self->getKtran());
            self->expansion.setSymmetry(self->getSymmetry());
            self->expansion.setPolarization(polarization);
        }
    } else if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->incidentVector(side, polarization, k0);
        self->expansion.setK0(k0);
        return 100. * self->getTransmission(incident, side);
    }, wavelength);
}


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
        } catch (py::error_already_set&) {
            PyErr_Clear();
            try {
                if (py::len(value) != 2) throw py::error_already_set();
                self.mirrors.reset(std::make_pair<double,double>(double(py::extract<double>(value[0])), double(py::extract<double>(value[1]))));
            } catch (py::error_already_set&) {
                throw ValueError("None, float, or tuple of two floats required");
            }
        }
    }
}


static py::object FourierSolver2D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);
    auto* expansion = &self->expansion;

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
        WHAT_NEFF,
        WHAT_KTRAN
    };
    What what = WHAT_NOTHING;
    py::object array;

    plask::optional<dcomplex> k0, neff, ktran;

    AxisNames* axes = getCurrentAxes();
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
        } else if (*i == "neff") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_NEFF; array = kwargs[*i];
            } else
                neff.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran()) {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_KTRAN; array = kwargs[*i];
            } else
                ktran.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "dispersive") {
            throw TypeError(u8"Dispersive argument has been removed: set solver.lam0 attribute");
        } else
            throw TypeError(u8"get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();

    if (k0) expansion->setK0(*k0); else expansion->setK0(self->getK0());
    if (neff) { if (what != WHAT_WAVELENGTH && what != WHAT_K0) expansion->setBeta(*neff * expansion->k0); }
    else expansion->setBeta(self->getBeta());
    if (ktran) expansion->setKtran(*ktran); else expansion->setKtran(self->getKtran());
    expansion->setLam0(self->getLam0());
    expansion->setSymmetry(self->getSymmetry());
    expansion->setPolarization(self->getPolarization());

    switch (what) {
        case WHAT_NOTHING:
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self, neff](dcomplex x) -> dcomplex { self->expansion.setK0(2e3*PI/x); if (neff) self->expansion.setBeta(*neff * self->expansion.k0); return self->getDeterminant(); },
                array
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self, neff](dcomplex x) -> dcomplex { self->expansion.setK0(x); if (neff) self->expansion.setBeta(*neff * x); return self->getDeterminant(); },
                array
            );
        case WHAT_NEFF:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setBeta(x * self->getK0()); return self->getDeterminant(); },
                array
            );
        case WHAT_KTRAN:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setKtran(x); return self->getDeterminant(); },
                array
            );
    }
    return py::object();
}

static size_t FourierSolver2D_setMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"set_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);
    auto* expansion = &self->expansion;

    AxisNames* axes = getCurrentAxes();
    plask::optional<dcomplex> k0, neff, ktran;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength") {
            if (k0) throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            k0.reset(2e3*PI / py::extract<dcomplex>(kwargs[*i])());
        } else if (*i == "k0") {
            if (k0) throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
            k0.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "neff") {
            neff.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran()) {
            ktran.reset(py::extract<dcomplex>(kwargs[*i]));
        } else
            throw TypeError(u8"set_mode() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();

    if (k0) expansion->setK0(*k0); else expansion->setK0(self->getK0());
    if (neff) expansion->setBeta(*neff * expansion->k0); else expansion->setBeta(self->getK0());
    if (ktran) expansion->setKtran(*ktran); else expansion->setKtran(self->getKtran());
    expansion->setLam0(self->getLam0());
    expansion->setSymmetry(self->getSymmetry());
    expansion->setPolarization(self->getPolarization());

    return self->setMode();
}

static size_t FourierSolver2D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"find_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError(u8"find_mode() takes exactly one keyword argument ({0} given)", py::len(kwargs));
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
        throw TypeError(u8"find_mode() got unexpected keyword argument '{0}'", key);

    return self->findMode(what, value);
}


static dcomplex FourierSolver2D_Mode_Neff(const FourierSolver2D::Mode& mode) {
    return mode.beta / mode.k0;
}

static py::object FourierSolver2D_Mode__getattr__(const FourierSolver2D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.beta);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError(u8"'Mode' object has no attribute '{0}'", name);
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
    return format(u8"<lam: {:.2f}nm, neff: {}, ktran: {}/um, polarization: {}, symmetry: {}, power: {:.2g} mW>",
                  real(2e3*PI / self.k0),
                  str(self.beta/self.k0, "{:.3f}{:+.3g}j"),
                  str(self.ktran, "({:.3g}{:+.3g}j)", "{:.3g}"),
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
    return format(u8"Fourier2D.Mode(lam={0}, neff={1}, ktran={2}, polarization={3}, symmetry={4}, power={5})",
                  str(2e3*PI/self.k0), str(self.beta/self.k0), str(self.ktran), pol, sym, self.power);
}

static py::object FourierSolver2D_getFieldVectorE(FourierSolver2D& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), self.separated(), 2);
}

static py::object FourierSolver2D_getFieldVectorH(FourierSolver2D& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec2D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), self.separated(), 2);
}

// static py::object FourierSolver2D_getReflectedFieldVectorE(FourierSolver2D::Reflected& self, double z) {
//     return arrayFromVec2D<NPY_CDOUBLE>(self.parent->getScatteredFieldVectorE(self.polarization, self.side, z), self.parent->separated(), 2);
// }
//
// static py::object FourierSolver2D_getReflectedFieldVectorH(FourierSolver2D::Reflected& self, double z) {
//     return arrayFromVec2D<NPY_CDOUBLE>(self.parent->getScatteredFieldVectorH(self.polarization, self.side, z), self.parent->separated(), 2);
// }



void export_FourierSolver2D()
{
    CLASS(FourierSolver2D, "Fourier2D",
        u8"Optical Solver using Fourier expansion in 2D.\n\n"
        u8"It calculates optical modes and optical field distribution using Fourier slab method\n"
        u8"and reflection transfer in two-dimensional Cartesian space.")
    export_base(solver);
    PROVIDER(outNeff, "Effective index of the last computed mode.");
    RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size.");
    RW_PROPERTY(symmetry, getSymmetry, setSymmetry, "Mode symmetry.");
    RW_PROPERTY(polarization, getPolarization, setPolarization, "Mode polarization.");
    solver.add_property("lam", &__Class__::getWavelength, &Solver_setWavelength<__Class__>,
                u8"Wavelength of the light [nm].\n\n"
                u8"Use this property only if you are looking for anything else than\n"
                u8"the wavelength, e.g. the effective index of lateral wavevector.\n");
    solver.add_property("wavelength", &__Class__::getWavelength, &Solver_setWavelength<__Class__>,
                u8"Alias for :attr:`lam`");
    solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>,
                u8"Normalized frequency of the light [1/µm].\n\n"
                u8"Use this property only if you are looking for anything else than\n"
                u8"the wavelength,e.g. the effective index of lateral wavevector.\n");
    RW_PROPERTY(klong, getBeta, setBeta,
                u8"Longitudinal propagation constant of the light [1/µm].\n\n"
                u8"Use this property only if you are looking for anything else than\n"
                u8"the longitudinal component of the propagation vector and the effective index.\n");
    RW_PROPERTY(ktran, getKtran, setKtran,
                u8"Transverse propagation constant of the light [1/µm].\n\n"
                u8"Use this property only  if you are looking for anything else than\n"
                u8"the transverse component of the propagation vector.\n");
    RW_FIELD(refine, "Number of refinement points for refractive index averaging.");
    RW_FIELD(oversampling, "Factor by which the number of coefficients is increased for FFT.");
    solver.add_property("dct", &__Class__::getDCT, &__Class__::setDCT,
                "Type of discrete cosine transform for symmetric expansion.");
    RW_FIELD(emission, "Direction of the useful light emission.\n\n"
                       u8"Necessary for the over-threshold model to correctly compute the output power.\n"
                       u8"Currently the fields are normalized only if this parameter is set to\n"
                       u8"``top`` or ``bottom``. Otherwise, it is ``undefined`` (default) and the fields\n"
                       u8"are not normalized.");
    solver.def("get_determinant", py::raw_function(FourierSolver2D_getDeterminant),
                u8"Compute discontinuity matrix determinant.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    neff (complex): Longitudinal effective index.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("find_mode", py::raw_function(FourierSolver2D_findMode),
                u8"Compute the mode near the specified effective index.\n\n"
                u8"Only one of the following arguments can be given through a keyword.\n"
                u8"It is the starting point for search of the specified parameter.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    neff (complex): Longitudinal effective index.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("set_mode", py::raw_function(FourierSolver2D_setMode),
                u8"Set the mode for specified parameters.\n\n"
                u8"This method should be used if you have found a mode manually and want to insert\n"
                u8"it into the solver in order to determine the fields. Calling this will raise an\n"
                u8"exception if the determinant for the specified parameters is too large.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    neff (complex): Longitudinal effective index.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("compute_reflectivity", &Solver_computeReflectivityOld<FourierSolver2D>, (py::arg("lam"), "side", "polarization"));      //TODO remove in the future
    solver.def("compute_transmittivity", &Solver_computeTransmittivityOld<FourierSolver2D>, (py::arg("lam"), "side", "polarization"));  //TODO remove in the future
    solver.def("compute_reflectivity", &Solver_computeReflectivity<FourierSolver2D>,
            u8"Compute reflection coefficient on planar incidence [%].\n\n"
            u8"Args:\n"
            u8"    lam (float or array of floats): Incident light wavelength.\n"
            u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
            u8"        present.\n"
            u8"    polarization: Specification of the incident light polarization.\n"
            u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
            u8"        name of the non-vanishing electric field component.\n"
            , (py::arg("lam"), "side", "polarization"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity<FourierSolver2D>,
            u8"Compute transmission coefficient on planar incidence [%].\n\n"
            u8"Args:\n"
            u8"    lam (float or array of floats): Incident light wavelength.\n"
            u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
            u8"        present.\n"
            u8"    polarization: Specification of the incident light polarization.\n"
            u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
            u8"        of the non-vanishing electric field component.\n"
            , (py::arg("lam"), "side", "polarization"));
    solver.add_property("mirrors", FourierSolver2D_getMirrors, FourierSolver2D_setMirrors,
                u8"Mirror reflectivities. If None then they are automatically estimated from the\n"
                u8"Fresnel equations.");
    solver.add_property("pml", py::make_function(&Solver_getPML<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setPML<FourierSolver2D>,
                        u8"Side Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC
                        );
    RO_FIELD(modes, "Computed modes.");
    solver.def("scattering", Scattering<FourierSolver2D>::get, py::with_custodian_and_ward_postcall<0,1>(),
               u8"Access to the reflected field.\n\n"
               u8"Args:\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               u8"        of the non-vanishing electric field component.\n\n"
               u8":rtype: Fourier2D.Scattering\n"
               , (py::arg("side"), "polarization")
              );
    solver.def("get_raw_E", FourierSolver2D_getFieldVectorE, (py::arg("num"), "level"),
               u8"Get Fourier expansion coefficients for the electric field.\n\n"
               u8"This is a low-level function returning :math:`E_l` and/or :math:`E_t` Fourier\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("get_raw_H", FourierSolver2D_getFieldVectorH, (py::arg("num"), "level"),
               u8"Get Fourier expansion coefficients for the magnetic field.\n\n"
               u8"This is a low-level function returning :math:`H_l` and/or :math:`H_t` Fourier\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("layer_eigenmodes", &Eigenmodes<FourierSolver2D>::init, py::arg("level"),
               u8"Get eignemodes for a layer at specified level.\n\n"
               u8"This is a low-level function to access diagonalized eigenmodes for a specific\n"
               u8"layer. Please refer to the detailed solver description for the interpretation\n"
               u8"of the returned values.\n\n"
               u8"Args:\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n",
               py::with_custodian_and_ward_postcall<0,1>()
              );
    // OBSOLETE
    solver.def("get_electric_coefficients", FourierSolver2D_getFieldVectorE, (py::arg("num"), "level"),
               u8"Obsolete alias for :meth:`get_raw_E`.");
    solver.def("get_magnetic_coefficients", FourierSolver2D_getFieldVectorH, (py::arg("num"), "level"),
               u8"Obsolete alias for :meth:`get_raw_H`.");

    py::scope scope = solver;
    (void) scope;   // don't warn about unused variable scope

    register_vector_of<FourierSolver2D::Mode>("Modes");
    py::class_<FourierSolver2D::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .def_readonly("symmetry", &FourierSolver2D::Mode::symmetry, u8"Mode horizontal symmetry.")
        .def_readonly("polarization", &FourierSolver2D::Mode::polarization, u8"Mode polarization.")
        .add_property("lam", &getModeWavelength<FourierSolver2D::Mode>, u8"Mode wavelength [nm].")
        .add_property("wavelength", &getModeWavelength<FourierSolver2D::Mode>, u8"Mode wavelength [nm].")
        .def_readonly("k0", &FourierSolver2D::Mode::k0, u8"Mode normalized frequency [1/µm].")
        .def_readonly("beta", &FourierSolver2D::Mode::beta, u8"Mode longitudinal wavevector [1/µm].")
        .add_property("neff", &FourierSolver2D_Mode_Neff, u8"Mode longitudinal effective index [-].")
        .def_readonly("ktran", &FourierSolver2D::Mode::ktran, u8"Mode transverse wavevector [1/µm].")
        .def_readwrite("power", &FourierSolver2D::Mode::power, u8"Total power emitted into the mode [mW].")
        .def("__str__", &FourierSolver2D_Mode_str)
        .def("__repr__", &FourierSolver2D_Mode_repr)
        .def("__getattr__", &FourierSolver2D_Mode__getattr__)
    ;

    Scattering<FourierSolver2D>::registerClass("2D");
    Eigenmodes<FourierSolver2D>::registerClass("2D");
}

}}}} // namespace plask::optical::slab::python
