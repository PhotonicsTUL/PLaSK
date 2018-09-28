#define PY_ARRAY_UNIQUE_SYMBOL PLASK_OPTICAL_SLAB_ARRAY_API
#define NO_IMPORT_ARRAY

#include "fourier3d-python.h"
#include "slab-python.h"

namespace plask { namespace optical { namespace slab { namespace python {


template <NPY_TYPES type>
static inline py::object arrayFromVec3D(cvector data, size_t minor, int dim) {
    npy_intp dims[] = { npy_intp(data.size()/(2*minor)), npy_intp(minor), 2 };
    npy_intp strides[] = { npy_intp(2*minor*sizeof(dcomplex)), npy_intp(2*sizeof(dcomplex)), npy_intp(sizeof(dcomplex)) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, type, strides, (void*)data.data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from field coefficients");
    PythonDataVector<const dcomplex,3> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}

template <>
py::object Eigenmodes<FourierSolver3D>::array(const dcomplex* data, size_t N) const {
    npy_intp dims[] = { npy_intp(N/(2*solver.minor())), npy_intp(solver.minor()), 2 };
    npy_intp strides[] = { npy_intp(2*solver.minor()*sizeof(dcomplex)), npy_intp(2*sizeof(dcomplex)), npy_intp(sizeof(dcomplex)) };
    PyObject* arr = PyArray_New(&PyArray_Type, 3, dims, NPY_CDOUBLE, strides, (void*)data, 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array");
    return py::object(py::handle<>(arr));
}


py::object FourierSolver3D_Mode__getattr__(const FourierSolver3D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.klong);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError(u8"'Mode' object has no attribute '{0}'", name);
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
    return format(u8"<lam: {}nm, klong: {}/um, ktran: {}/um, symmetry: ({}), power: {:.2g}mW>",
                  str(2e3*PI/self.k0, u8"({:.3f}{:+.3g}j)", "{:.3f}"),
                  str(self.klong, u8"{:.3f}{:+.3g}j", "{:.3f}"),
                  str(self.ktran, u8"{:.3f}{:+.3g}j", "{:.3f}"),
                  FourierSolver3D_Mode_symmetry(self),
                  self.power
                 );
}
std::string FourierSolver3D_Mode_repr(const FourierSolver3D::Mode& self) {
    return format(u8"Fourier3D.Mode(lam={0}, klong={1}, ktran={2}, symmetry=({3}), power={4})",
                  str(2e3*PI/self.k0),
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
            default: throw IndexError(u8"index out of range");
        }
    }

    void __setitem__(int i, const typename WrappedType<T>::Wrapper& value) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: *ptr_long = value; self->invalidate(); return;
            case 1: *ptr_tran = value; self->invalidate(); return;
            default: throw IndexError(u8"index out of range");
        }
    }

    typename WrappedType<T>::Wrapper __getattr__(const std::string& name) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) return WrappedType<T>::make(self, ptr_long);
        if (name == "tran" || name == "t" || name == axes->getNameForTran()) return WrappedType<T>::make(self, ptr_tran);
        throw AttributeError(u8"object has no attribute '{0}'", name);
    }

    void __setattr__(const std::string& name, const typename WrappedType<T>::Wrapper& value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) { *ptr_long = value; self->invalidate(); }
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) { *ptr_tran = value; self->invalidate(); }
        else throw AttributeError(u8"object has no attribute '{0}'", name);
    }

    std::string __str__() {
        return "(" + std::string(py::extract<std::string>(py::str(py::object(WrappedType<T>::make(self, ptr_long))))) + ", "
                   + std::string(py::extract<std::string>(py::str(py::object(WrappedType<T>::make(self, ptr_tran))))) + ")";
    }

    static void register_(const std::string& name) {
        py::class_<FourierSolver3D_LongTranWrapper<T>>(name.c_str(), u8"Access wrapper for parameter along long/tran axis", py::no_init)
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
        } catch (py::error_already_set&) {
            PyErr_Clear();
            try {
                FourierSolver3D_LongTranWrapper<T>* value = py::extract<FourierSolver3D_LongTranWrapper<T>*>(object);
                self.*field_long = *value->ptr_long;
                self.*field_tran = *value->ptr_tran;
                self.invalidate();
            } catch (py::error_already_set&) {
                PyErr_Clear();
                try {
                    if (py::len(object) != 2) throw py::error_already_set();
                    T value_long = py::extract<T>(object[0]),
                    value_tran = py::extract<T>(object[1]);
                    self.*field_long = value_long;
                    self.*field_tran = value_tran;
                    self.invalidate();
                } catch (py::error_already_set&) {
                    throw TypeError(u8"You may only assign a value or a sequence of two values");
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
            default: throw IndexError(u8"index out of range");
        }
    }

    void __setitem__(int i, Expansion::Component value) {
        if (i < 0) i = 2 - i;
        switch (i) {
            case 0: self->setSymmetryLong(value); return;
            case 1: self->setSymmetryTran(value); return;
            default: throw IndexError(u8"index out of range");
        }
    }

    Expansion::Component __getattr__(const std::string& name) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) return self->getSymmetryLong();
        if (name == "tran" || name == "t" || name == axes->getNameForTran()) return self->getSymmetryTran();
        throw AttributeError(u8"object has no attribute '{0}'", name);
    }

    void __setattr__(const std::string& name, Expansion::Component value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) self->setSymmetryLong(value);
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) self->setSymmetryTran(value);
        else throw AttributeError(u8"object has no attribute '{0}'", name);
    }

    std::string __str__() {
        return "(" + std::string(py::extract<std::string>(py::str(py::object(self->getSymmetryLong())))) + ", "
                   + std::string(py::extract<std::string>(py::str(py::object(self->getSymmetryTran())))) + ")";
    }

    static void register_() {
        py::class_<FourierSolver3D_SymmetryLongTranWrapper>("Symmetries", u8"Access wrapper for parameter along long/tran axis", py::no_init)
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
        catch (py::error_already_set&) { throw TypeError(u8"You may only assign a sequence of two values"); }
        self.setSymmetryLong(py::extract<Expansion::Component>(values[0]));
        self.setSymmetryTran(py::extract<Expansion::Component>(values[1]));
    }
};



py::object FourierSolver3D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError(u8"get_determinant() takes exactly one non-keyword argument ({0} given)", py::len(args));
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
    plask::optional<dcomplex> wavelength, k0;

    AxisNames* axes = getCurrentAxes();
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                wavelength.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                k0.reset(dcomplex(py::extract<dcomplex>(kwargs[*i])));
        } else if (*i == "klong" || *i == "kl" || *i == "k"+axes->getNameForLong()) {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_KLONG; array = kwargs[*i];
            } else
                klong = py::extract<dcomplex>(kwargs[*i]);
        } else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran()) {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError(u8"Only one key may be an array");
                what = WHAT_KTRAN; array = kwargs[*i];
            } else
                ktran = py::extract<dcomplex>(kwargs[*i]);
        } else if (*i == "dispersive") {
            throw TypeError(u8"Dispersive argument has been removed: set solver.lam0 attribute");
        } else
            throw TypeError(u8"get_determinant() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();

    if (wavelength) {
        if (k0) throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
        expansion->setK0(2e3*PI / (*wavelength));
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
                [self](dcomplex x) -> dcomplex { self->expansion.setK0(2e3*PI/x); return self->getDeterminant(); },
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
        throw TypeError(u8"set_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    plask::optional<dcomplex> wavelength, k0;
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
            throw TypeError(u8"set_mode() got unexpected keyword argument '{0}'", *i);
    }

    self->Solver::initCalculation();

    if (wavelength) {
        if (k0) throw BadInput(self->getId(), u8"'lam' and 'k0' are mutually exclusive");
        self->expansion.setK0(2e3*PI / (*wavelength));
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
        throw TypeError(u8"find_mode() takes exactly one non-keyword argument ({0} given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError(u8"find_mode() takes exactly one keyword argument ({0} given)", py::len(kwargs));
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
        throw TypeError(u8"find_mode() got unexpected keyword argument '{0}'", key);

    return self->findMode(what, value);
}


static py::object FourierSolver3D_getFieldVectorE(FourierSolver3D& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec3D<NPY_CDOUBLE>(self.getFieldVectorE(std::size_t(num), z), self.minor(), 3);
}

static py::object FourierSolver3D_getFieldVectorH(FourierSolver3D& self, int num, double z) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return arrayFromVec3D<NPY_CDOUBLE>(self.getFieldVectorH(std::size_t(num), z), self.minor(), 3);
}



void export_FourierSolver3D()
{
    CLASS(FourierSolver3D, "Fourier3D",
        u8"Optical Solver using Fourier expansion in 3D.\n\n"
        u8"It calculates optical modes and optical field distribution using Fourier slab method\n"
        u8"and reflection transfer in three-dimensional Cartesian space.")
    export_base(solver);
    solver.add_property("size",
                        py::make_function(FourierSolver3D_getSize, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<size_t>(&FourierSolver3D::size_long, &FourierSolver3D::size_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        u8"Orthogonal expansion sizes in longitudinal and transverse directions.");
    solver.add_property("refine",
                        py::make_function(FourierSolver3D_getRefine, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<size_t>(&FourierSolver3D::refine_long, &FourierSolver3D::refine_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        u8"Number of refinement points for refractive index averaging in longitudinal and transverse directions.");
    solver.add_property("oversampling",
                        py::make_function(FourierSolver3D_getOversampling, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<double>(&FourierSolver3D::oversampling_long, &FourierSolver3D::oversampling_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        u8"Factor by which the number of coefficients is increased for FFT.");
    solver.add_property("pmls",
                        py::make_function(FourierSolver3D_getPml, py::with_custodian_and_ward_postcall<0,1>()),
                        py::make_function(FourierSolver3D_LongTranSetter<PML>(&FourierSolver3D::pml_long, &FourierSolver3D::pml_tran),
                                            py::default_call_policies(),
                                            boost::mpl::vector3<void, FourierSolver3D&, py::object>()),
                        u8"Longitudinal and transverse edge Perfectly Matched Layers boundary conditions.\n\n"
                        PML_ATTRS_DOC);
    solver.add_property("symmetry",
                        py::make_function(&FourierSolver3D_SymmetryLongTranWrapper::getter, py::with_custodian_and_ward_postcall<0,1>()),
                        &FourierSolver3D_SymmetryLongTranWrapper::setter,
                        u8"Longitudinal and transverse mode symmetries.\n");
    solver.add_property("dct", &__Class__::getDCT, &__Class__::setDCT, "Type of discrete cosine transform for symmetric expansion.");
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
    RW_PROPERTY(klong, getKlong, setKlong,
                u8"Longitudinal propagation constant of the light [1/µm].\n\n"
                u8"Use this property only if you are looking for anything else than\n"
                u8"the longitudinal component of the propagation vector and the effective index.\n");
    RW_PROPERTY(ktran, getKtran, setKtran,
                u8"Transverse propagation constant of the light [1/µm].\n\n"
                u8"Use this property only  if you are looking for anything else than\n"
                u8"the transverse component of the propagation vector.\n");
    RW_FIELD(emission, u8"Direction of the useful light emission.\n\n"
                       u8"Necessary for the over-threshold model to correctly compute the output power.\n"
                       u8"Currently the fields are normalized only if this parameter is set to\n"
                       u8"``top`` or ``bottom``. Otherwise, it is ``undefined`` (default) and the fields\n"
                       u8"are not normalized.");
    solver.def("get_determinant", py::raw_function(FourierSolver3D_getDeterminant),
                u8"Compute discontinuity matrix determinant.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    klong (complex): Longitudinal wavevector.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("find_mode", py::raw_function(FourierSolver3D_findMode),
                u8"Compute the mode near the specified effective index.\n\n"
                u8"Only one of the following arguments can be given through a keyword.\n"
                u8"It is the starting point for search of the specified parameter.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    klong (complex): Longitudinal wavevector.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("set_mode", py::raw_function(FourierSolver3D_setMode),
                u8"Set the mode for specified parameters.\n\n"
                u8"This method should be used if you have found a mode manually and want to insert\n"
                u8"it into the solver in order to determine the fields. Calling this will raise an\n"
                u8"exception if the determinant for the specified parameters is too large.\n\n"
                u8"Arguments can be given through keywords only.\n\n"
                u8"Args:\n"
                u8"    lam (complex): Wavelength.\n"
                u8"    k0 (complex): Normalized frequency.\n"
                u8"    klong (complex): Longitudinal wavevector.\n"
                u8"    ktran (complex): Transverse wavevector.\n");
    solver.def("compute_reflectivity", &Solver_computeReflectivity_polarization<FourierSolver3D>,
               (py::arg("lam"), "side", "polarization"));
    solver.def("compute_reflectivity", &Solver_computeReflectivity_index<FourierSolver3D>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_reflectivity", &Solver_computeReflectivity_array<FourierSolver3D>,
               (py::arg("lam"), "side", "coffs"),
               u8"Compute reflection coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
               u8"        name of the non-vanishing electric field component.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_polarization<FourierSolver3D>,
               (py::arg("lam"), "side", "polarization"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_index<FourierSolver3D>,
               (py::arg("lam"), "side", "index"));
    solver.def("compute_transmittivity", &Solver_computeTransmittivity_array<FourierSolver3D>,
               (py::arg("lam"), "side", "coffs"),
               u8"Compute transmission coefficient on planar incidence [%].\n\n"
               u8"Args:\n"
               u8"    lam (float or array of floats): Incident light wavelength.\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               u8"        of the non-vanishing electric field component.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n");
    solver.def("scattering", Scattering<FourierSolver3D>::from_polarization, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "polarization"));
    solver.def("scattering", Scattering<FourierSolver3D>::from_index, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "idx"));
    solver.def("scattering", Scattering<FourierSolver3D>::from_array, py::with_custodian_and_ward_postcall<0,1>(), (py::arg("side"), "coeffs"),
               u8"Access to the reflected field.\n\n"
               u8"Args:\n"
               u8"    side (`top` or `bottom`): Side of the structure where the incident light is\n"
               u8"        present.\n"
               u8"    polarization: Specification of the incident light polarization.\n"
               u8"        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
               u8"        of the non-vanishing electric field component.\n"
               u8"    idx: Eigenmode number.\n"
               u8"    coeffs: expansion coefficients of the incident vector.\n\n"
               u8":rtype: Fourier3D.Scattering\n"
              );
    solver.def("get_raw_E", FourierSolver3D_getFieldVectorE, (py::arg("num"), "level"),
               u8"Get Fourier expansion coefficients for the electric field.\n\n"
               u8"This is a low-level function returning :math:`E_l` and/or :math:`E_t` Fourier\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("get_raw_H", FourierSolver3D_getFieldVectorH, (py::arg("num"), "level"),
               u8"Get Fourier expansion coefficients for the magnetic field.\n\n"
               u8"This is a low-level function returning :math:`H_l` and/or :math:`H_t` Fourier\n"
               u8"expansion coefficients. Please refer to the detailed solver description for their\n"
               u8"interpretation.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
               u8":rtype: numpy.ndarray\n"
              );
    solver.def("layer_eigenmodes", &Eigenmodes<FourierSolver3D>::fromZ, py::arg("level"),
               u8"Get eignemodes for a layer at specified level.\n\n"
               u8"This is a low-level function to access diagonalized eigenmodes for a specific\n"
               u8"layer. Please refer to the detailed solver description for the interpretation\n"
               u8"of the returned values.\n\n"
               u8"Args:\n"
               u8"    level (float): Vertical level at which the coefficients are computed.\n",
               py::with_custodian_and_ward_postcall<0,1>()
              );
    RO_FIELD(modes, "Computed modes.");

    // OBSOLETE
    solver.def("get_electric_coefficients", FourierSolver3D_getFieldVectorE, (py::arg("num"), "level"),
               u8"Obsolete alias for :meth:`get_raw_E`.");
    solver.def("get_magnetic_coefficients", FourierSolver3D_getFieldVectorH, (py::arg("num"), "level"),
               u8"Obsolete alias for :meth:`get_raw_H`.");

    py::scope scope = solver;
    (void) scope;   // don't warn about unused variable scope

    register_vector_of<FourierSolver3D::Mode>("Modes");
    py::class_<FourierSolver3D::Mode>("Mode", u8"Detailed information about the mode.", py::no_init)
        .add_property("symmetry", &FourierSolver3D_Mode_symmetry, u8"Mode horizontal symmetry.")
        .add_property("lam", &getModeWavelength<FourierSolver3D::Mode>, u8"Mode wavelength [nm].")
        .add_property("wavelength", &getModeWavelength<FourierSolver3D::Mode>, u8"Mode wavelength [nm].")
        .def_readonly("k0", &FourierSolver3D::Mode::k0, u8"Mode normalized frequency [1/µm].")
        .def_readonly("klong", &FourierSolver3D::Mode::klong, u8"Mode longitudinal wavevector [1/µm].")
        .def_readonly("ktran", &FourierSolver3D::Mode::ktran, u8"Mode transverse wavevector [1/µm].")
        .def_readwrite("power", &FourierSolver3D::Mode::power, u8"Total power emitted into the mode [mW].")
        .def("__str__", &FourierSolver3D_Mode_str)
        .def("__repr__", &FourierSolver3D_Mode_repr)
        .def("__getattr__", &FourierSolver3D_Mode__getattr__)
    ;

    Scattering<FourierSolver3D>::registerClass("3D");
    Eigenmodes<FourierSolver3D>::registerClass("3D");

    FourierSolver3D_LongTranWrapper<size_t>::register_("Sizes");
    FourierSolver3D_LongTranWrapper<double>::register_("Oversampling");
    FourierSolver3D_LongTranWrapper<PML>::register_("PMLs");
    FourierSolver3D_SymmetryLongTranWrapper::register_();
}

}}}} // namespace plask::optical::slab::python

