#include "fourier3d-python.h"
#include "slab-python.h"

namespace plask { namespace solvers { namespace slab { namespace python {


template <NPY_TYPES type>
static inline py::object arrayFromVec3D(cvector data, size_t minor, int dim) {
    npy_intp dims[] = { data.size()/(2*minor), minor, 2 };
    npy_intp strides[] = { 2*minor*sizeof(dcomplex), 2*sizeof(dcomplex), sizeof(dcomplex) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, type, strides, (void*)data.data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from field coefficients");
    DataVectorWrap<const dcomplex,3> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}


py::object FourierSolver3D_Mode__getattr__(const FourierSolver3D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.klong);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError("'Mode' object has no attribute '{0}'", name);
    return py::object();
}


std::string FourierSolver3D_Mode_symmetry(const FourierSolver3D::Mode& self) {
    AxisNames* axes = getCurrentAxes();
    std::string syml, symt;
    switch (self.symmetry_long) {
        case Expansion::E_TRAN: syml = "E" + axes->getNameForTran(); break;
        case Expansion::E_LONG: syml = "E" + axes->getNameForLong(); break;
        default: syml = "none";
    }
    switch (self.symmetry_tran) {
        case Expansion::E_TRAN: symt = "E" + axes->getNameForTran(); break;
        case Expansion::E_LONG: symt = "E" + axes->getNameForLong(); break;
        default: symt = "none";
    }
    return syml + "," + symt;
}

std::string FourierSolver3D_Mode_str(const FourierSolver3D::Mode& self) {
    return format("<lam: {}nm, klong: {}/um, ktran: {}/um, symmetry: ({}), power: {:.2g}mW>",
                  str(2e3*M_PI/self.k0, "({:.3f}{:+.3g}j)", "{:.3f}"),
                  str(self.klong, "{:.3f}{:+.3g}j", "{:.3f}"),
                  str(self.ktran, "{:.3f}{:+.3g}j", "{:.3f}"),
                  FourierSolver3D_Mode_symmetry(self),
                  self.power
                 );
}
std::string FourierSolver3D_Mode_repr(const FourierSolver3D::Mode& self) {
    return format("Fourier3D.Mode(lam={0}, klong={1}, ktran={2}, symmetry=({3}), power={4})",
                  str(2e3*M_PI/self.k0),
                  str(self.klong),
                  str(self.ktran),
                  FourierSolver3D_Mode_symmetry(self),
                  self.power
                 );
}


template <typename T>
struct FourierSolver3D_LongTranWrapper {
    FourierSolver3D* self;
    T* ptr_long;
    T* ptr_tran;

    FourierSolver3D_LongTranWrapper(FourierSolver3D* self, T* ln, T* tr):
        self(self), ptr_long(ln), ptr_tran(tr) {}

    typename WrappedType<T>::Wrapper __getitem__(int i) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: return WrappedType<T>::make(self, ptr_long);
            case 1: return WrappedType<T>::make(self, ptr_tran);
            default: throw IndexError("index out of range");
        }
    }

    void __setitem__(int i, const typename WrappedType<T>::Wrapper& value) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: *ptr_long = value; self->invalidate(); return;
            case 1: *ptr_tran = value; self->invalidate(); return;
            default: throw IndexError("index out of range");
        }
    }

    typename WrappedType<T>::Wrapper __getattr__(const std::string& name) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) return WrappedType<T>::make(self, ptr_long);
        if (name == "tran" || name == "t" || name == axes->getNameForTran()) return WrappedType<T>::make(self, ptr_tran);
        throw AttributeError("object has no attribute '{0}'", name);
    }

    void __setattr__(const std::string& name, const typename WrappedType<T>::Wrapper& value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) { *ptr_long = value; self->invalidate(); }
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) { *ptr_tran = value; self->invalidate(); }
        else throw AttributeError("object has no attribute '{0}'", name);
    }

    std::string __str__() {
        return "(" + std::string(py::extract<std::string>(py::str(py::object(WrappedType<T>::make(self, ptr_long))))) + ", "
                   + std::string(py::extract<std::string>(py::str(py::object(WrappedType<T>::make(self, ptr_tran))))) + ")";
    }

    static void register_(const std::string& name) {
        py::class_<FourierSolver3D_LongTranWrapper<T>>(name.c_str(), "Access wrapper for parameter along long/tran axis", py::no_init)
            .def("__getitem__", &FourierSolver3D_LongTranWrapper<T>::__getitem__, typename WrappedType<T>::CallPolicy())
            .def("__setitem__", &FourierSolver3D_LongTranWrapper<T>::__setitem__)
            .def("__getattr__", &FourierSolver3D_LongTranWrapper<T>::__getattr__, typename WrappedType<T>::CallPolicy())
            .def("__setattr__", &FourierSolver3D_LongTranWrapper<T>::__setattr__)
            .def("__str__", &FourierSolver3D_LongTranWrapper<T>::__str__)
        ;
    }
};

template <typename T>
struct FourierSolver3D_LongTranSetter {
    T FourierSolver3D::* field_long;
    T FourierSolver3D::* field_tran;

    FourierSolver3D_LongTranSetter(T FourierSolver3D::* ln, T FourierSolver3D::* tr):
        field_long(ln), field_tran(tr) {}

    void operator()(FourierSolver3D& self, const py::object object) {
        try {
            typename WrappedType<T>::Extracted value = py::extract<typename WrappedType<T>::Extracted>(object);
            self.*field_long = value;
            self.*field_tran = value;
            self.invalidate();
        } catch (py::error_already_set) {
            PyErr_Clear();
            try {
                FourierSolver3D_LongTranWrapper<T>* value = py::extract<FourierSolver3D_LongTranWrapper<T>*>(object);
                self.*field_long = *value->ptr_long;
                self.*field_tran = *value->ptr_tran;
                self.invalidate();
            } catch (py::error_already_set) {
                PyErr_Clear();
                try {
                    if (py::len(object) != 2) throw py::error_already_set();
                    T value_long = py::extract<T>(object[0]),
                    value_tran = py::extract<T>(object[1]);
                    self.*field_long = value_long;
                    self.*field_tran = value_tran;
                    self.invalidate();
                } catch (py::error_already_set) {
                    throw TypeError("You may only assign a value or a sequence of two values");
                }
            }
        }
    }
};

FourierSolver3D_LongTranWrapper<size_t> FourierSolver3D_getSize(FourierSolver3D* self) {
    return FourierSolver3D_LongTranWrapper<size_t>(self, &self->size_long, &self->size_tran);
}

FourierSolver3D_LongTranWrapper<size_t> FourierSolver3D_getRefine(FourierSolver3D* self) {
    return FourierSolver3D_LongTranWrapper<size_t>(self, &self->refine_long, &self->refine_tran);
}

FourierSolver3D_LongTranWrapper<double> FourierSolver3D_getOversampling(FourierSolver3D* self) {
    return FourierSolver3D_LongTranWrapper<double>(self, &self->oversampling_long, &self->oversampling_tran);
}

FourierSolver3D_LongTranWrapper<PML> FourierSolver3D_getPml(FourierSolver3D* self) {
    return FourierSolver3D_LongTranWrapper<PML>(self, &self->pml_long, &self->pml_tran);
}

struct FourierSolver3D_SymmetryLongTranWrapper {
    FourierSolver3D* self;

    FourierSolver3D_SymmetryLongTranWrapper(FourierSolver3D* self): self(self) {}

    Expansion::Component __getitem__(int i) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: return self->getSymmetryLong();
            case 1: return self->getSymmetryTran();
            default: throw IndexError("index out of range");
        }
    }

    void __setitem__(int i, Expansion::Component value) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: self->setSymmetryLong(value); return;
            case 1: self->setSymmetryTran(value); return;
            default: throw IndexError("index out of range");
        }
    }

    Expansion::Component __getattr__(const std::string& name) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) return self->getSymmetryLong();
        if (name == "tran" || name == "t" || name == axes->getNameForTran()) return self->getSymmetryTran();
        throw AttributeError("object has no attribute '{0}'", name);
    }

    void __setattr__(const std::string& name, Expansion::Component value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) self->setSymmetryLong(value);
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) self->setSymmetryTran(value);
        else throw AttributeError("object has no attribute '{0}'", name);
    }

    std::string __str__() {
        return "(" + std::string(py::extract<std::string>(py::str(py::object(self->getSymmetryLong())))) + ", "
                   + std::string(py::extract<std::string>(py::str(py::object(self->getSymmetryTran())))) + ")";
    }

    static void register_() {
        py::class_<FourierSolver3D_SymmetryLongTranWrapper>("Symmetries", "Access wrapper for parameter along long/tran axis", py::no_init)
            .def("__getitem__", &FourierSolver3D_SymmetryLongTranWrapper::__getitem__)
            .def("__setitem__", &FourierSolver3D_SymmetryLongTranWrapper::__setitem__)
            .def("__getattr__", &FourierSolver3D_SymmetryLongTranWrapper::__getattr__)
            .def("__setattr__", &FourierSolver3D_SymmetryLongTranWrapper::__setattr__)
            .def("__str__", &FourierSolver3D_SymmetryLongTranWrapper::__str__)
        ;
    }

    static FourierSolver3D_SymmetryLongTranWrapper getter(FourierSolver3D* self) {
        return FourierSolver3D_SymmetryLongTranWrapper(self);
    }

    static void setter(FourierSolver3D& self, py::object values) {
        try { if (py::len(values) != 2 || py::extract<std::string>(values).check()) throw py::error_already_set(); }
        catch (py::error_already_set) { throw TypeError("You may only assign a sequence of two values"); }
        self.setSymmetryLong(py::extract<Expansion::Component>(values[0]));
        self.setSymmetryTran(py::extract<Expansion::Component>(values[1]));
    }
};



py::object FourierSolver3D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);
    auto* expansion = &self->expansion;

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
        WHAT_KLONG,
        WHAT_KTRAN
    };
    What what = WHAT_NOTHING;
    py::object array;

    dcomplex klong = self->getKlong(), ktran = self->getKtran();
    boost::optional<dcomplex> wavelength, k0;

    AxisNames* axes = getCurrentAxes();
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                wavelength.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                k0.reset(dcomplex(py::extract<dcomplex>(kwargs[*i])));
        } else if (*i == "klong" || *i == "kl" || *i == "k"+axes->getNameForLong()) {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_KLONG; array = kwargs[*i];
            } else
                klong = py::extract<dcomplex>(kwargs[*i]);
        } else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran()) {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_KTRAN; array = kwargs[*i];
            } else
                ktran = py::extract<dcomplex>(kwargs[*i]);
        } else if (*i == "dispersive") {
            throw TypeError("Dispersive argument has been removed: set solver.lam0 attribute");
        } else
            throw TypeError("get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    self->initCalculation();

    if (wavelength) {
        if (k0) throw BadInput(self->getId(), "'lam' and 'k0' are mutually exclusive");
        expansion->setK0(2e3*M_PI / (*wavelength));
    } else if (k0)
        expansion->setK0(*k0);
    else
        expansion->setK0(self->getK0());

    expansion->setKlong(klong);
    expansion->setKtran(ktran);
    expansion->setLam0(self->getLam0());
    expansion->setSymmetryLong(self->getSymmetryLong());
    expansion->setSymmetryTran(self->getSymmetryTran());

    switch (what) {
        case WHAT_NOTHING:
            return py::object(self->getDeterminant());
        case WHAT_WAVELENGTH:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setK0(2e3*M_PI/x); return self->getDeterminant(); },
                array
            );
        case WHAT_K0:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setK0(x); return self->getDeterminant(); },
                array
            );
        case WHAT_KLONG:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->expansion.setKlong(x); return self->getDeterminant(); },
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

static size_t FourierSolver3D_setMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("set_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    boost::optional<dcomplex> wavelength, k0;
    dcomplex klong = self->getKlong(), ktran = self->getKtran();

    AxisNames* axes = getCurrentAxes();
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam") {
            wavelength.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0")
            k0.reset(dcomplex(py::extract<dcomplex>(kwargs[*i])));
        else if (*i == "klong" || *i == "kl" || *i == "k"+axes->getNameForLong())
            klong = py::extract<dcomplex>(kwargs[*i]);
        else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran())
            ktran = py::extract<dcomplex>(kwargs[*i]);
        else
            throw TypeError("set_mode() got unexpected keyword argument '{0}'", *i);
    }

    self->initCalculation();

    if (wavelength) {
        if (k0) throw BadInput(self->getId(), "'lam' and 'k0' are mutually exclusive");
        self->expansion.setK0(2e3*M_PI / (*wavelength));
    } else if (k0)
        self->expansion.setK0(*k0);
    else
        self->expansion.setK0(self->getK0());

    self->expansion.setKlong(klong);
    self->expansion.setKtran(ktran);
    self->expansion.setLam0(self->getLam0());
    self->expansion.setSymmetryLong(self->getSymmetryLong());
    self->expansion.setSymmetryTran(self->getSymmetryTran());

    return self->setMode();
}

size_t FourierSolver3D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("find_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError("find_mode() takes exactly one keyword argument ({0} given)", py::len(kwargs));
    std::string key = py::extract<std::string>(kwargs.keys()[0]);
    dcomplex value = py::extract<dcomplex>(kwargs[key]);
    AxisNames* axes = getCurrentAxes();
    FourierSolver3D::What what;

    if (key == "lam")
        what = FourierSolver3D::WHAT_WAVELENGTH;
    else if (key == "k0")
        what = FourierSolver3D::WHAT_K0;
    else if (key == "klong" || key == "kl" || key == "k"+axes->getNameForLong())
        what = FourierSolver3D::WHAT_KLONG;
    else if (key == "ktran" || key == "kt" || key == "k"+axes->getNameForTran())
        what = FourierSolver3D::WHAT_KTRAN;
    else
        throw TypeError("find_mode() got unexpected keyword argument '{0}'", key);

    return self->findMode(what, value);
}

static py::object FourierSolver3D_reflectedAmplitudes(FourierSolver3D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    if (!self.initCalculation()) {
        self.setExpansionDefaults(false);
        self.expansion.setK0(2e3*M_PI/lam);
    } else
        self.expansion.setK0(2e3*M_PI/lam);
    auto data = self.getReflectedAmplitudes(polarization, incidence);
    return arrayFromVec3D<NPY_DOUBLE>(data, self.minor(), 2);
}

static py::object FourierSolver3D_transmittedAmplitudes(FourierSolver3D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    if (!self.initCalculation()) {
        self.setExpansionDefaults(false);
        self.expansion.setK0(2e3*M_PI/lam);
    } else
        self.expansion.setK0(2e3*M_PI/lam);
    auto data = self.getTransmittedAmplitudes(polarization, incidence);
    return arrayFromVec3D<NPY_DOUBLE>(data, self.minor(), 2);
}

static py::object FourierSolver3D_reflectedCoefficients(FourierSolver3D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    if (!self.initCalculation()) {
        self.setExpansionDefaults(false);
        self.expansion.setK0(2e3*M_PI/lam);
    } else
        self.expansion.setK0(2e3*M_PI/lam);
    auto data = self.getReflectedCoefficients(polarization, incidence);
    return arrayFromVec3D<NPY_CDOUBLE>(data, self.minor(), 3);
}

static py::object FourierSolver3D_transmittedCoefficients(FourierSolver3D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    if (!self.initCalculation()) {
        self.expansion.setK0(2e3*M_PI/lam);
        self.setExpansionDefaults(false);
    } else
        self.expansion.setK0(2e3*M_PI/lam);
    auto data = self.getTransmittedCoefficients(polarization, incidence);
    return arrayFromVec3D<NPY_CDOUBLE>(data, self.minor(), 3);
}

static py::object FourierSolver3D_getFieldVectorE(FourierSolver3D& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number {:d}", num);
    return arrayFromVec3D<NPY_CDOUBLE>(self.getFieldVectorE(num, z), self.minor(), 3);
}

static py::object FourierSolver3D_getFieldVectorH(FourierSolver3D& self, int num, double z) {
    if (num < 0) num = self.modes.size() + num;
    if (num >= self.modes.size()) throw IndexError("Bad mode number {:d}", num);
    return arrayFromVec3D<NPY_CDOUBLE>(self.getFieldVectorH(num, z), self.minor(), 3);
}

static py::object FourierSolver3D_getReflectedFieldVectorE(FourierSolver3D::Reflected& self, double z) {
    return arrayFromVec3D<NPY_CDOUBLE>(self.parent->getReflectedFieldVectorE(self.polarization, self.side, z), self.parent->minor(), 3);
}

static py::object FourierSolver3D_getReflectedFieldVectorH(FourierSolver3D::Reflected& self, double z) {
    return arrayFromVec3D<NPY_CDOUBLE>(self.parent->getReflectedFieldVectorH(self.polarization, self.side, z), self.parent->minor(), 3);
}



void export_FourierSolver3D()
{
    plask_import_array();

    CLASS(FourierSolver3D, "Fourier3D",
        "Optical Solver using Fourier expansion in 3D.\n\n"
        "It calculates optical modes and optical field distribution using Fourier slab method\n"
        "and reflection transfer in three-dimensional Cartesian space.")
    export_base(solver);
    solver.add_property("size",
                        py::make_function(FourierSolver3D_getSize, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<size_t>(&FourierSolver3D::size_long, &FourierSolver3D::size_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        "Orthogonal expansion sizes in longitudinal and transverse directions.");
    solver.add_property("refine",
                        py::make_function(FourierSolver3D_getRefine, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<size_t>(&FourierSolver3D::refine_long, &FourierSolver3D::refine_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        "Number of refinement points for refractive index averaging in longitudinal and transverse directions.");
    solver.add_property("oversampling",
                        py::make_function(FourierSolver3D_getOversampling, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<double>(&FourierSolver3D::oversampling_long, &FourierSolver3D::oversampling_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        "Factor by which the number of coefficients is increased for FFT.");
    solver.add_property("pmls",
                        py::make_function(FourierSolver3D_getPml, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<PML>(&FourierSolver3D::pml_long, &FourierSolver3D::pml_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        "Longitudinal and transverse edge Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC);
    solver.add_property("symmetry",
                        py::make_function(&FourierSolver3D_SymmetryLongTranWrapper::getter, py::with_custodian_and_ward_postcall<0,1>()),
                        &FourierSolver3D_SymmetryLongTranWrapper::setter,
                        "Longitudinal and transverse mode symmetries.\n");
    solver.add_property("dct", &__Class__::getDCT, &__Class__::setDCT, "Type of discrete cosine transform for symmetric expansion.");
    solver.add_property("lam", &__Class__::getWavelength, &Solver_setWavelength<__Class__>,
                "Wavelength of the light [nm].\n\n"
                "Use this property only if you are looking for anything else than\n"
                "the wavelength, e.g. the effective index of lateral wavevector.\n");
    solver.add_property("wavelength", &__Class__::getWavelength, &Solver_setWavelength<__Class__>,
                "Alias for :attr:`lam`");
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
    RW_FIELD(emission, "Direction of the useful light emission.\n\n"
                       "Necessary for the over-threshold model to correctly compute the output power.\n"
                       "Currently the fields are normalized only if this parameter is set to\n"
                       "``top`` or ``bottom``. Otherwise, it is ``undefined`` (default) and the fields\n"
                       "are not normalized.");
    solver.def("get_determinant", py::raw_function(FourierSolver3D_getDeterminant),
                "Compute discontinuity matrix determinant.\n\n"
                "Arguments can be given through keywords only.\n\n"
                "Args:\n"
                "    lam (complex): Wavelength.\n"
                "    k0 (complex): Normalized frequency.\n"
                "    klong (complex): Longitudinal wavevector.\n"
                "    ktran (complex): Transverse wavevector.\n");
    solver.def("find_mode", py::raw_function(FourierSolver3D_findMode),
                "Compute the mode near the specified effective index.\n\n"
                "Only one of the following arguments can be given through a keyword.\n"
                "It is the starting point for search of the specified parameter.\n\n"
                "Args:\n"
                "    lam (complex): Wavelength.\n"
                "    k0 (complex): Normalized frequency.\n"
                "    klong (complex): Longitudinal wavevector.\n"
                "    ktran (complex): Transverse wavevector.\n");
    solver.def("set_mode", py::raw_function(FourierSolver3D_setMode),
                "Set the mode for specified parameters.\n\n"
                "This method should be used if you have found a mode manually and want to insert\n"
                "it into the solver in order to determine the fields. Calling this will raise an\n"
                "exception if the determinant for the specified parameters is too large.\n\n"
                "Arguments can be given through keywords only.\n\n"
                "Args:\n"
                "    lam (complex): Wavelength.\n"
                "    k0 (complex): Normalized frequency.\n"
                "    klong (complex): Longitudinal wavevector.\n"
                "    ktran (complex): Transverse wavevector.\n");
    solver.def("compute_reflectivity", &Solver_computeReflectivity<FourierSolver3D>,
            "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
            "Args:\n"
            "    lam (float or array of floats): Incident light wavelength.\n"
            "    polarization: Specification of the incident light polarization.\n"
            "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
            "        name of the non-vanishing electric field component.\n"
            "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
            "        present.\n"
            , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity<FourierSolver3D>,
            "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
            "Args:\n"
            "    lam (float or array of floats): Incident light wavelength.\n"
            "    polarization: Specification of the incident light polarization.\n"
            "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
            "        of the non-vanishing electric field component.\n"
            "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
            "        present.\n"
            , (py::arg("lam"), "polarization", "side"));
    solver.def("reflected", FourierSolver_getReflected<FourierSolver3D>, py::with_custodian_and_ward_postcall<0,1>(),
               "Access to the reflected field.\n\n"
               "Args:\n"
               "    lam (float): Incident light wavelength.\n"
               "    polarization: Specification of the incident light polarization.\n"
               "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               "        of the non-vanishing electric field component.\n"
               "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               "        present.\n\n"
               ":rtype: Fourier3D.Reflected\n"
               , (py::arg("lam"), "polarization", "side")
              );
    solver.def("compute_reflected_orders", &FourierSolver3D_reflectedAmplitudes,
                "Compute amplitudes of all the Fourier coefficients (diffraction orders)\n"
                "of the reflected field [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_transmitted_orders", &FourierSolver3D_transmittedAmplitudes,
                "Compute amplitudes of all the Fourier coefficients (diffraction orders)\n"
                "of the transmited field [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_reflected_coefficients", &FourierSolver3D_reflectedCoefficients,
                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("compute_transmitted_coefficients", &FourierSolver3D_transmittedCoefficients,
                "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
                "Args:\n"
                "    lam (float): Incident light wavelength.\n"
                "    polarization: Specification of the incident light polarization.\n"
                "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                "        name of the non-vanishing electric field component.\n"
                "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                "        present.\n"
                , (py::arg("lam"), "polarization", "side"));
    solver.def("get_electric_coefficients", FourierSolver3D_getFieldVectorE, (py::arg("num"), "level"),
               "Get Fourier expansion coefficients for the electric field.\n\n"
               "This is a low-level function returning :math:`E_l` and/or :math:`E_t` Fourier\n"
               "expansion coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical level at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
    solver.def("get_magnetic_coefficients", FourierSolver3D_getFieldVectorH, (py::arg("num"), "level"),
               "Get Fourier expansion coefficients for the magnetic field.\n\n"
               "This is a low-level function returning :math:`H_l` and/or :math:`H_t` Fourier\n"
               "expansion coefficients. Please refer to the detailed solver description for their\n"
               "interpretation.\n\n"
               "Args:\n"
               "    num (int): Computed mode number.\n"
               "    level (float): Vertical level at which the coefficients are computed.\n\n"
               ":rtype: numpy.ndarray\n"
              );
    // solver.add_property("material_mesh_long", &__Class__::getLongMesh,
    //                     "Regular mesh with points in which material is sampled along longitudinal direction.");
    // solver.add_property("material_mesh_tran", &__Class__::getTranMesh,
    //                     "Regular mesh with points in which material is sampled along transverse direction.");
    RO_FIELD(modes, "Computed modes.");
    py::scope scope = solver;

    register_vector_of<FourierSolver3D::Mode>("Modes");
    py::class_<FourierSolver3D::Mode>("Mode", "Detailed information about the mode.", py::no_init)
        .add_property("symmetry", &FourierSolver3D_Mode_symmetry, "Mode horizontal symmetry.")
        .add_property("lam", &getModeWavelength<FourierSolver3D::Mode>, "Mode wavelength [nm].")
        .add_property("wavelength", &getModeWavelength<FourierSolver3D::Mode>, "Mode wavelength [nm].")
        .def_readonly("k0", &FourierSolver3D::Mode::k0, "Mode normalized frequency [1/µm].")
        .def_readonly("klong", &FourierSolver3D::Mode::klong, "Mode longitudinal wavevector [1/µm].")
        .def_readonly("ktran", &FourierSolver3D::Mode::ktran, "Mode transverse wavevector [1/µm].")
        .def_readwrite("power", &FourierSolver3D::Mode::power, "Total power emitted into the mode [mW].")
        .def("__str__", &FourierSolver3D_Mode_str)
        .def("__repr__", &FourierSolver3D_Mode_repr)
        .def("__getattr__", &FourierSolver3D_Mode__getattr__)
    ;


    py::class_<FourierSolver3D::Reflected, shared_ptr<FourierSolver3D::Reflected>, boost::noncopyable>("Reflected",
        "Reflected mode proxy.\n\n"
        "This class contains providers for the optical field for a reflected field"
        "under the normal incidence.\n"
        , py::no_init)
        .def_readonly("outElectricField", reinterpret_cast<ProviderFor<LightE,Geometry3D> FourierSolver3D::Reflected::*>
                                            (&FourierSolver3D::Reflected::outElectricField),
            format(docstring_attr_provider<LightE>(), "LightE", "3D", "electric field", "V/m", "", "", "", "outElectricField").c_str()
        )
        .def_readonly("outMagneticField", reinterpret_cast<ProviderFor<LightH,Geometry3D> FourierSolver3D::Reflected::*>
                                            (&FourierSolver3D::Reflected::outMagneticField),
            format(docstring_attr_provider<LightH>(), "LightH", "3D", "magnetic field", "A/m", "", "", "", "outMagneticField").c_str()
        )
        .def_readonly("outLightMagnitude", reinterpret_cast<ProviderFor<LightMagnitude,Geometry3D> FourierSolver3D::Reflected::*>
                                            (&FourierSolver3D::Reflected::outLightMagnitude),
            format(docstring_attr_provider<LightMagnitude>(), "LightMagnitude", "3D", "light intensity", "W/m²", "", "", "", "outLightMagnitude").c_str()
        )
        .def("get_electric_coefficients", FourierSolver3D_getReflectedFieldVectorE, py::arg("level"),
             "Get Fourier expansion coefficients for the electric field.\n\n"
             "This is a low-level function returning :math:`E_l` and/or :math:`E_t` Fourier\n"
             "expansion coefficients. Please refer to the detailed solver description for their\n"
             "interpretation.\n\n"
             "Args:\n"
             "    level (float): Vertical level at which the coefficients are computed.\n\n"
             ":rtype: numpy.ndarray\n"
            )
        .def("get_magnetic_coefficients", FourierSolver3D_getReflectedFieldVectorH, py::arg("level"),
             "Get Fourier expansion coefficients for the magnegtic field.\n\n"
             "This is a low-level function returning :math:`H_l` and/or :math:`H_t` Fourier\n"
             "expansion coefficients. Please refer to the detailed solver description for their\n"
             "interpretation.\n\n"
             "Args:\n"
             "    level (float): Vertical level at which the coefficients are computed.\n\n"
             ":rtype: numpy.ndarray\n"
            )
    ;

    py_enum<FourierSolver3D::Emission>()
        .value("UNDEFINED", FourierSolver3D::EMISSION_UNSPECIFIED)
        .value("TOP", FourierSolver3D::EMISSION_TOP)
        .value("BOTTOM", FourierSolver3D::EMISSION_BOTTOM)
    ;

    FourierSolver3D_LongTranWrapper<size_t>::register_("Sizes");
    FourierSolver3D_LongTranWrapper<double>::register_("Oversampling");
    FourierSolver3D_LongTranWrapper<PML>::register_("PMLs");
    FourierSolver3D_SymmetryLongTranWrapper::register_();
}

}}}} // namespace plask::solvers::slab::python
