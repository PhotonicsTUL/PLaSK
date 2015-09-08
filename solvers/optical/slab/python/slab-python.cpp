/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
#include <boost/python/raw_function.hpp>
using namespace plask;
using namespace plask::python;

#if defined(_WIN32) && defined(interface)
#   undef interface
#endif

#include "../fourier/solver2d.h"
#include "../fourier/solver3d.h"
#include "../bessel/solvercyl.h"
using namespace plask::solvers::slab;

#define ROOTDIGGER_ATTRS_DOC \
    ".. rubric:: Attributes:\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.slab.RootParams.alpha\n" \
    "   ~optical.slab.RootParams.lambd\n" \
    "   ~optical.slab.RootParams.initial_range\n" \
    "   ~optical.slab.RootParams.maxiter\n" \
    "   ~optical.slab.RootParams.maxstep\n" \
    "   ~optical.slab.RootParams.method\n" \
    "   ~optical.slab.RootParams.tolf_max\n" \
    "   ~optical.slab.RootParams.tolf_min\n" \
    "   ~optical.slab.RootParams.tolx\n\n" \
    ":rtype: RootParams\n"

#define PML_ATTRS_DOC \
    ".. rubric:: Attributes:\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.slab.PML.factor\n" \
    "   ~optical.slab.PML.shape\n" \
    "   ~optical.slab.PML.dist\n" \
    "   ~optical.slab.PML.size\n\n" \
    ":rtype: PML"

template <typename SolverT>
static py::object Solver_getLam0(const SolverT& self) {
    if (self.lam0) return *self.lam0;
    else return py::object();
}
    
template <typename SolverT>
static void Solver_setLam0(SolverT& self, py::object value) {
    if (value == py::object()) self.clearLam0();
    else self.setLam0(py::extract<double>(value));
}
    
template <typename SolverT>
static py::tuple SlabSolver_getStack(const SolverT& self) {
    py::list result;
    for (auto i: self.getStack()) {
        result.append(i);
    }
    return py::tuple(result);
}

template <typename SolverT>
static py::tuple SlabSolver_getLayerSets(const SolverT& self) {
    py::list result;
    for (auto i: self.getLayersPoints()) {
        result.append(i);
    }
    return py::tuple(result);
}

struct PythonComponentConventer {

    // Determine if obj can be converted into an Aligner
    static void* convertible(PyObject* obj) {
        if (PyString_Check(obj) || obj == Py_None) return obj;
        return nullptr;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<Expansion::Component>*)data)->storage.bytes;
        AxisNames* axes = getCurrentAxes();
        Expansion::Component val;
        if (obj == Py_None) {
            val = Expansion::E_UNSPECIFIED;
        } else {
            try {
                std::string repr = py::extract<std::string>(obj);
                if (repr == "none" || repr == "NONE" || repr == "None")
                    val = Expansion::E_UNSPECIFIED;
                else if (repr == "Etran" || repr == "Et" || repr == "E"+axes->getNameForTran() ||
                         repr == "Hlong" || repr == "Hl" || repr == "H"+axes->getNameForLong())
                    val = Expansion::E_TRAN;
                else if (repr == "Elong" || repr == "El" || repr == "E"+axes->getNameForLong() ||
                         repr == "Htran" || repr == "Ht" || repr == "H"+axes->getNameForTran())
                    val = Expansion::E_LONG;
                else
                    throw py::error_already_set();
            } catch (py::error_already_set) {
                throw ValueError("Wrong component specification.");
            }
        }
        new(storage) Expansion::Component(val);
        data->convertible = storage;
    }

    static PyObject* convert(Expansion::Component val) {
        AxisNames* axes = getCurrentAxes();
        switch (val) {
            case Expansion::E_TRAN: return py::incref(py::object("E"+axes->getNameForTran()).ptr());
            case Expansion::E_LONG: return py::incref(py::object("E"+axes->getNameForLong()).ptr());
            default: return py::incref(Py_None);
        }
    }
};

struct PmlWrapper {

    Solver* solver;
    PML* pml;

    PmlWrapper(Solver* solver, PML* pml): solver(solver), pml(pml) {}

    PmlWrapper(const PmlWrapper& orig): solver(orig.solver) {
        if (solver) pml = orig.pml;
        else pml = new PML(*pml);
    }

    ~PmlWrapper() { if (!solver) delete pml; }

    static shared_ptr<PmlWrapper> __init__(dcomplex factor, double size, double dist, double order) {
        return make_shared<PmlWrapper>(nullptr, new PML(factor, size, dist, order));
    }

    operator PML() const { return *pml; }

    dcomplex get_factor() const { return pml->factor; }
    void set_factor(dcomplex val) {
        pml->factor = val;
        if (solver) solver->invalidate();
    }

    double get_size() const { return pml->size; }
    void set_size(double val) {
        pml->size = val;
        if (solver) solver->invalidate();
    }

    double get_dist() const { return pml->dist; }
    void set_dist(double val) {
        pml->dist = val;
        if (solver) solver->invalidate();
    }

    double get_order() const { return pml->order; }
    void set_order(double val) {
        pml->order = val;
        if (solver) solver->invalidate();
    }

    std::string __str__() const {
        return format("<factor: %1%, size: %2%, dist: %3%, shape: %4%>", str(pml->factor), pml->size, pml->dist, pml->order);
    }

    std::string __repr__() const {
        return format("PML(factor=%1%, size=%2%, dist=%3%, shape=%4%)", str(pml->factor), pml->size, pml->dist, pml->order);
    }
};

template <typename T>
struct WrappedType {
    typedef T Wrapper;
    typedef T Extracted;
    typedef py::default_call_policies CallPolicy;
    template <typename S> static Wrapper make(S* solver, T* item) { return *item; }
};

template <>
struct WrappedType<PML> {
    typedef PmlWrapper Wrapper;
    typedef PmlWrapper& Extracted;
    typedef py::with_custodian_and_ward_postcall<0,1> CallPolicy;
    template <typename S> static Wrapper make(S* solver, PML* item) { return PmlWrapper(solver, item); }
};


template <typename SolverT>
void Solver_setWavelength(SolverT& self, dcomplex lam) { self.setWavelength(lam); }

template <typename SolverT>
void Solver_setK0(SolverT& self, dcomplex k0) { self.setWavelength(k0); }

template <typename SolverT>
PmlWrapper Solver_vPML(SolverT* self) {
    return PmlWrapper(self, &self->vpml);
}

template <typename SolverT>
void Solver_setvPML(SolverT* self, const PmlWrapper& value) {
    self->vpml = *value.pml;
    self->invalidate();
}

#ifndef NDEBUG
struct CMatrix_Python {
    cmatrix data;
    CMatrix_Python(const cmatrix& data): data(data) {}
    CMatrix_Python(const CMatrix_Python& src): data(src.data) {}
    
    static PyObject* convert(const cmatrix& self) {
        npy_intp dims[2] = { self.rows(), self.cols() };
        npy_intp strides[2] = {sizeof(dcomplex), self.rows() * sizeof(dcomplex)};
        
        PyObject* arr = PyArray_New(&PyArray_Type, 2, dims, NPY_CDOUBLE, strides,
                                    (void*)self.data(), 0, 0, NULL);
        if (arr == nullptr) throw plask::CriticalException("Cannot create array from matrix");
        // Make sure the data vector stays alive as long as the array
        py::object oself {CMatrix_Python(self)};
        py::incref(oself.ptr());        
        PyArray_SetBaseObject((PyArrayObject*)arr, oself.ptr());
        return arr;
    }
};
#endif


py::object FourierSolver2D_getMirrors(const FourierSolver2D& self) {
    if (!self.mirrors) return py::object();
    return py::make_tuple(self.mirrors->first, self.mirrors->second);
}

void FourierSolver2D_setMirrors(FourierSolver2D& self, py::object value) {
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


py::object FourierSolver2D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("get_determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
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
            throw TypeError("Dispersive argument has been removes. Set solver.lam0 attribute.");
        else
            throw TypeError("get_determinant() got unexpected keyword argument '%1%'", *i);
    }
    
    FourierSolver2D::ParamGuard guard(self);

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

size_t FourierSolver2D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("find_mode() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError("find_mode() takes exactly one keyword argument (%1% given)", py::len(kwargs));
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
        throw TypeError("find_mode() got unexpected keyword argument '%1%'", key);

    return self->findMode(what, value);
}


template <typename SolverT>
py::object FourierSolver_computeReflectivity(SolverT* self,
                                             py::object wavelength,
                                             Expansion::Component polarization,
                                             Transfer::IncidentDirection incidence
                                            )
{
    typename SolverT::ParamGuard guard(self);
    return UFUNC<double>([=](double lam)->double {
        self->setWavelength(lam);
        return 100. * self->getReflection(polarization, incidence);
    }, wavelength);
}

template <typename SolverT>
py::object FourierSolver_computeTransmittivity(SolverT* self,
                                               py::object wavelength,
                                               Expansion::Component polarization,
                                               Transfer::IncidentDirection incidence
                                              )
{
    typename SolverT::ParamGuard guard(self);
    return UFUNC<double>([=](double lam)->double {
        self->setWavelength(lam);
        return 100. * self->getTransmission(polarization, incidence);
    }, wavelength);
}

template <NPY_TYPES type>
static inline py::object arrayFromVec2D(cvector data, bool sep) {
    int strid = sep? 1 : 2;
    npy_intp dims[] = { data.size() / strid };
    npy_intp strides[] = { strid * sizeof(dcomplex) };
    PyObject* arr = PyArray_New(&PyArray_Type, 1, dims, type, strides, (void*)data.data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from field coefficients");
    DataVectorWrap<const dcomplex,2> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}

py::object FourierSolver2D_reflectedAmplitudes(FourierSolver2D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    FourierSolver2D::ParamGuard guard(&self);
    self.setWavelength(lam);
    auto data = self.getReflectedAmplitudes(polarization, incidence);
    return arrayFromVec2D<NPY_DOUBLE>(data, self.separated());
}

py::object FourierSolver2D_transmittedAmplitudes(FourierSolver2D& self, double lam, Expansion::Component polarization, Transfer::IncidentDirection incidence) {
    FourierSolver2D::ParamGuard guard(&self);
    self.setWavelength(lam);
    auto data = self.getTransmittedAmplitudes(polarization, incidence);
    return arrayFromVec2D<NPY_DOUBLE>(data, self.separated());
}


template <typename Solver>
static PmlWrapper Solver_getPML(Solver* self) {
    return PmlWrapper(self, &self->pml);
}

template <typename Solver>
static void Solver_setPML(Solver* self, const PmlWrapper& value) {
    self->pml = *value.pml;
    self->invalidate();
}

template <typename SolverT>
shared_ptr<typename SolverT::Reflected> FourierSolver_getReflected(SolverT* parent,
                                                                  double wavelength,
                                                                  Expansion::Component polarization,
                                                                  Transfer::IncidentDirection side)
{
    return make_shared<typename SolverT::Reflected>(parent, wavelength, polarization, side);
}

dcomplex FourierSolver2D_Mode_Neff(const FourierSolver2D::Mode& mode) {
    return mode.beta / mode.k0;
}

py::object FourierSolver2D_Mode__getattr__(const FourierSolver2D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.beta);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError("'Mode' object has no attribute '%1%'", name);
    return py::object();
}


template <typename Mode>
dcomplex getModeWavelength(const Mode& mode) {
    return 2e3 * M_PI / mode.k0;
}

std::string FourierSolver2D_Mode_str(const FourierSolver2D::Mode& self) {
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
    return format("<lam: %.2fnm, neff: %.3f%+.3g, ktran: %s/um, polarization: %s, symmetry: %s, power: %.2g mW>",
                  real(2e3*M_PI / self.k0),
                  real(neff),imag(neff),
                  (imag(self.ktran) == 0.)? format("%.3g",real(self.ktran)) : format("%.3g%+.3g",real(self.ktran),imag(self.ktran)),
                  pol,
                  sym,
                  self.power
                 );
}
std::string FourierSolver2D_Mode_repr(const FourierSolver2D::Mode& self) {
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
    return format("Fourier2D.Mode(lam=%1%, neff=%2%, ktran=%3%, polarization=%4%, symmetry=%5%, power=%6%)",
                  str(2e3*M_PI/self.k0), str(self.beta/self.k0), str(self.ktran), pol, sym, self.power);
}


std::string BesselSolverCyl_Mode_str(const BesselSolverCyl::Mode& self) {
    dcomplex lam = 2e3*M_PI / self.k0;
    return format("<m: %d, lam: (%.3f%+.3gj) nm, power: %.2g mW>", self.m, lam.real(), lam.imag(), self.power);
}
std::string BesselSolverCyl_Mode_repr(const BesselSolverCyl::Mode& self) {
    return format("BesselCyl.Mode(m=%d, lam=%g, power=%g)", self.m, str(2e3*M_PI / self.k0), self.power);
}

py::object BesselSolverCyl_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("get_determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    BesselSolverCyl* self = py::extract<BesselSolverCyl*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
    };
    What what = WHAT_NOTHING;
    py::object array;
    int m = 1;

    boost::optional<dcomplex> lambda;
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
        else if (*i == "dispersive")
            throw TypeError("Dispersive argument has been removes. Set solver.lam0 attribute.");
        else if (*i == "m")
            m = py::extract<int>(kwargs[*i]);
        else
            throw TypeError("get_determinant() got unexpected keyword argument '%1%'", *i);
    }
    if (lambda) self->setWavelength(*lambda);

    BesselSolverCyl::ParamGuard guard(self);

    self->setM(m);
    
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
    }
    return py::object();
}

py::object FourierSolver3D_Mode__getattr__(const FourierSolver3D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.klong);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError("'Mode' object has no attribute '%1%'", name);
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
    dcomplex lam = 2e3*M_PI / self.k0;
    return format("<lam: (%.3f%+.3gj) nm, klong: %s/um, ktran: %s/um, symmetry: (%s), power: %.2gmW>",
                  real(lam), imag(lam),
                  (imag(self.klong) == 0.)? format("%.3g",real(self.klong)) : format("%.3g%+.3g",real(self.klong),imag(self.klong)),
                  (imag(self.ktran) == 0.)? format("%.3g",real(self.ktran)) : format("%.3g%+.3g",real(self.ktran),imag(self.ktran)),
                  FourierSolver3D_Mode_symmetry(self),
                  self.power
                 );
}
std::string FourierSolver3D_Mode_repr(const FourierSolver3D::Mode& self) {
    return format("Fourier3D.Mode(lam=%1%, klong=%2%, ktran=%3%, symmetry=(%4%), power=%5%)",
                  str(2e3*M_PI / self.k0),
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
        throw AttributeError("object has no attribute '%1%'", name);
    }

    void __setattr__(const std::string& name, const typename WrappedType<T>::Wrapper& value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) { *ptr_long = value; self->invalidate(); }
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) { *ptr_tran = value; self->invalidate(); }
        else throw AttributeError("object has no attribute '%1%'", name);
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
        throw AttributeError("object has no attribute '%1%'", name);
    }

    void __setattr__(const std::string& name, Expansion::Component value) {
        AxisNames* axes = getCurrentAxes();
        if (name == "long" || name == "l" || name == axes->getNameForLong()) self->setSymmetryLong(value);
        else if (name == "tran" || name == "t" || name == axes->getNameForLong()) self->setSymmetryTran(value);
        else throw AttributeError("object has no attribute '%1%'", name);
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
        throw TypeError("get_determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    enum What {
        WHAT_NOTHING = 0,
        WHAT_WAVELENGTH,
        WHAT_K0,
        WHAT_KLONG,
        WHAT_KTRAN
    };
    What what = WHAT_NOTHING;
    py::object array;

    AxisNames* axes = getCurrentAxes();
    py::stl_input_iterator<std::string> begin(kwargs), end;
    boost::optional<dcomplex> wavelength, k0;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam") {
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_WAVELENGTH; array = kwargs[*i];
            } else
                wavelength.reset(py::extract<dcomplex>(kwargs[*i]));
        } else if (*i == "k0")
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_K0; array = kwargs[*i];
            } else
                k0.reset(dcomplex(py::extract<dcomplex>(kwargs[*i])));
        else if (*i == "klong" || *i == "kl" || *i == "k"+axes->getNameForLong())
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_KLONG; array = kwargs[*i];
            } else
                self->setKlong(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran())
            if (PyArray_Check(py::object(kwargs[*i]).ptr())) {
                if (what) throw TypeError("Only one key may be an array");
                what = WHAT_KTRAN; array = kwargs[*i];
            } else
                self->setKtran(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "dispersive")
            throw TypeError("Dispersive argument has been removes. Set solver.lam0 attribute.");
        else
            throw TypeError("get_determinant() got unexpected keyword argument '%1%'", *i);
    }
    
    FourierSolver3D::ParamGuard guard(self);
            
    if (wavelength) self->setWavelength(*wavelength);
    if (k0) self->setK0(*k0);

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
        case WHAT_KLONG:
            return UFUNC<dcomplex>(
                [self](dcomplex x) -> dcomplex { self->setKlong(x); return self->getDeterminant(); },
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

size_t FourierSolver3D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("find_mode() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError("find_mode() takes exactly one keyword argument (%1% given)", py::len(kwargs));
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
        throw TypeError("find_mode() got unexpected keyword argument '%1%'", key);

    return self->findMode(what, value);
}

#ifndef NDEBUG
template <typename Solver>
py::tuple Solver_getMatrices(Solver& self, size_t layer) {
    cmatrix RE, RH;
    self.getMatrices(layer, RE, RH);
    return py::make_tuple(py::object(RE), py::object(RH));
}
#endif


template <typename Class>
inline void export_base(Class solver) {
    typedef typename Class::wrapped_type Solver;
    solver.def_readwrite("outdist", &Solver::outdist, "Distance outside outer borders where material is sampled.");
    solver.add_property("interface", &Solver::getInterface, &Solver::setInterface, "Matching interface position.");
    solver.def("set_interface", (void(Solver::*)(const shared_ptr<GeometryObject>&, const PathHints&))&Solver::setInterfaceOn,
               "Set interface at the bottom of the specified object.\n\n"
               "Args:\n"
               "    object (geometry object): object to set the interface at.\n"
               "    path (path): Optional path specifying an instance of the object.",
               (py::arg("object"), py::arg("path")=py::object()));
    solver.def("set_interface", &Solver::setInterfaceAt,
               "Set interface as close as possible to the specified position.\n\n"
               "Args:\n"
               "    pos (float): Position, near which the interface will be located.", py::arg("pos"));
    solver.def_readwrite("smooth", &Solver::smooth, "Smoothing parameter for material boundaries (increases convergence).");
    solver.add_property("stack", &SlabSolver_getStack<Solver>, "Stack of distinct layers.");
    solver.add_property("layer_sets", &SlabSolver_getLayerSets<Solver>, "Vertical positions of layers in each layer set.");
    solver.add_property("group_layers", &Solver::getGroupLayers, &Solver::setGroupLayers,
                        "Layer grouping switch.\n\n"
                        "If this property is ``True``, similar layers are grouped for efficiency.");
    solver.template add_receiver<ReceiverFor<Temperature, typename Solver::SpaceType>, Solver>("inTemperature", &Solver::inTemperature, "");
    solver.template add_receiver<ReceiverFor<Gain, typename Solver::SpaceType>, Solver>("inGain", &Solver::inGain, "");
    solver.add_provider("outRefractiveIndex", &Solver::outRefractiveIndex, "");
    solver.add_provider("outLightMagnitude", &Solver::outLightMagnitude, "");
    solver.add_provider("outElectricField", &Solver::outElectricField, "");
    solver.add_provider("outMagneticField", &Solver::outMagneticField, "");
    solver.def_readwrite("root", &Solver::root,
                         "Configuration of the root searching algorithm.\n\n"
                         ROOTDIGGER_ATTRS_DOC
                        );
    solver.add_property("vpml", py::make_function(&Solver_vPML<Solver>, py::with_custodian_and_ward_postcall<0,1>()),
                        &Solver_setvPML<Solver>,
                        "Vertical Perfectly Matched Layers boundary conditions.\n\n"
                        ".. rubric:: Attributes\n\n"
                        ".. autosummary::\n\n"
                        "   ~optical.slab.PML.factor\n"
                        "   ~optical.slab.PML.dist\n"
                        "   ~optical.slab.PML.size\n\n"
                        "Attribute ``shape`` is ignored for vertical PML (it is always 0).\n"
                       );
    solver.add_property("transfer", &Solver::getTransferMethod, &Solver::setTransferMethod,
                        "Preferred transfer method.\n\n"
                        "Can take on of the following values:\n\n"
                        "============ ====================================\n"
                        "*auto*       Automatically choose the best method\n"
                        "*reflection* Reflection Transfer Method\n"
                        "*admittance* Admittance Transfer Method\n"
                        "============ ====================================\n"
                       );
    solver.add_property("lam0", Solver_setLam0<Solver>, Solver_setLam0<Solver>,
                        "Reference wavelength.\n\n"
                        "This is a wavelength at which refractive index is retrieved from the structure.\n"
                        "If this parameter is None, material parameters are computed each time,\n"
                        "the wavelenght changes even slightly (this is most accurate, but can be very\n"
                        "inefficient.\n"
                       );
    solver.def_readwrite("update_gain", &Solver::always_recompute_gain,
                        "Always update gain.\n\n"
                        "If this attribute is set to True, material parameters are always recomputed for\n"
                        "layers with gains. This allows to set py:attr:`lam0` for better efficiency and\n"
                        "still update gain for slight changes of wavelength.\n"
                       );
#ifndef NDEBUG
    solver.def("get_matrices", Solver_getMatrices<Solver>);
#endif
}

BOOST_PYTHON_MODULE(slab)
{
    plask_import_array();

#ifndef NDEBUG
    py::class_<CMatrix_Python>("_cmatrix", py::no_init);
    py::delattr(py::scope(), "_cmatrix");
    py::to_python_converter<cmatrix, CMatrix_Python>();
#endif
    
    py::to_python_converter<Expansion::Component, PythonComponentConventer>();
    py::converter::registry::push_back(&PythonComponentConventer::convertible, &PythonComponentConventer::construct,
                                       py::type_id<Expansion::Component>());

    // py::converter::registry::push_back(&PythonFourierSolver3DWhatConverter::convertible, &PythonFourierSolver3DWhatConverter::construct,
    //                                    py::type_id<FourierSolver3D::What>());

    py::class_<PmlWrapper, shared_ptr<PmlWrapper>>("PML", "Perfectly matched layer details.", py::no_init)
        .def("__init__", py::make_constructor(&PmlWrapper::__init__, py::default_call_policies(),
                                              (py::arg("factor"), "size", "dist", py::arg("shape")=2)))
        .add_property("factor", &PmlWrapper::get_factor, &PmlWrapper::set_factor, "PML scaling factor.")
        .add_property("size", &PmlWrapper::get_size, &PmlWrapper::set_size, "PML size.")
        .add_property("dist", &PmlWrapper::get_dist, &PmlWrapper::set_dist, "PML distance from the structure.")
        .add_property("shape", &PmlWrapper::get_order, &PmlWrapper::set_order, "PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.).")
        .def("__str__", &PmlWrapper::__str__)
        .def("__repr__", &PmlWrapper::__repr__)
    ;

    py_enum<Transfer::Method>()
        .value("AUTO", Transfer::METHOD_AUTO)
        .value("REFLECTION", Transfer::METHOD_REFLECTION)
        .value("ADMITTANCE", Transfer::METHOD_ADMITTANCE)
    ;

    py_enum<RootDigger::Method>()
        .value("MULLER", RootDigger::ROOT_MULLER)
        .value("BROYDEN", RootDigger::ROOT_BROYDEN)
        .value("BRENT", RootDigger::ROOT_BRENT)
    ;

    py_enum<typename ReflectionTransfer::IncidentDirection>()
        .value("TOP", ReflectionTransfer::INCIDENCE_TOP)
        .value("BOTTOM", ReflectionTransfer::INCIDENCE_BOTTOM)
    ;

    py::class_<RootDigger::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("method", &RootDigger::Params::method, "Root finding method ('muller', 'broyden',  or 'brent')")
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, "Maximum number of iterations.")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration (Broyden method only).")
        .def_readwrite("alpha", &RootDigger::Params::maxstep, "Parameter ensuring sufficient decrease of determinant in each step\n(Broyden method only).")
        .def_readwrite("lambd", &RootDigger::Params::maxstep, "Minimum decrease ratio of one step (Broyden method only).")
        .def_readwrite("initial_range", &RootDigger::Params::initial_dist, "Initial range size (Muller and Brent methods only).")
    ;

    {CLASS(FourierSolver2D, "Fourier2D",
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
        solver.def("reflected", &FourierSolver_getReflected<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>(),
                   "Access to the reflected field.\n\n"
                   "Args:\n"
                   "    lam (float): Incident light wavelength.\n"
                   "    polarization: Specification of the incident light polarization.\n"
                   "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                   "        of the non-vanishing electric field component.\n"
                   "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                   "        present.\n\n"
                   ":rtype: Fourier2D.Reflected\n"
                   , (py::arg("lam"), "polarization", "side"));
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
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightE", "2D", "electric field", "V/m", "", "", "", "outElectricField").c_str()
            )
            .def_readonly("outMagneticField", reinterpret_cast<ProviderFor<LightH,Geometry2DCartesian> FourierSolver2D::Reflected::*>
                                              (&FourierSolver2D::Reflected::outMagneticField),
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightH", "2D", "magnetic field", "A/m", "", "", "", "outMagneticField").c_str()
            )
            .def_readonly("outLightMagnitude", reinterpret_cast<ProviderFor<LightMagnitude,Geometry2DCartesian> FourierSolver2D::Reflected::*>
                                              (&FourierSolver2D::Reflected::outLightMagnitude),
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightMagnitude", "2D", "light intensity", "W/m²", "", "", "", "outLightMagnitude").c_str()
            )
        ;
    }

    {CLASS(FourierSolver3D, "Fourier3D",
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
         solver.def("compute_reflectivity", &FourierSolver_computeReflectivity<FourierSolver3D>,
                    "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
                    "Args:\n"
                    "    lam (float or array of floats): Incident light wavelength.\n"
                    "    polarization: Specification of the incident light polarization.\n"
                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                    "        name of the non-vanishing electric field component.\n"
                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                    "        present.\n"
                    , (py::arg("lam"), "polarization", "side"));
         solver.def("compute_transmittivity", &FourierSolver_computeTransmittivity<FourierSolver3D>,
                    "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
                    "Args:\n"
                    "    lam (float or array of floats): Incident light wavelength.\n"
                    "    polarization: Specification of the incident light polarization.\n"
                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                    "        of the non-vanishing electric field component.\n"
                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                    "        present.\n"
                    , (py::arg("lam"), "polarization", "side"));
         solver.def("reflected", &FourierSolver_getReflected<FourierSolver3D>, py::with_custodian_and_ward_postcall<0,1>(),
                    "Access to the reflected field.\n\n"
                    "Args:\n"
                    "    lam (float): Incident light wavelength.\n"
                    "    polarization: Specification of the incident light polarization.\n"
                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                    "        of the non-vanishing electric field component.\n"
                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                    "        present.\n\n"
                    ":rtype: Fourier3D.Reflected\n"
                    , (py::arg("lam"), "polarization", "side"));
        solver.add_property("material_mesh_long", &__Class__::getLongMesh,
                            "Regular mesh with points in which material is sampled along longitudinal direction.");
        solver.add_property("material_mesh_tran", &__Class__::getTranMesh,
                            "Regular mesh with points in which material is sampled along transverse direction.");
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
            .def_readwrite("power", &FourierSolver3D::Mode::power, "Total power emitted into the mode.")
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
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightE", "3D", "electric field", "V/m", "", "", "", "outElectricField").c_str()
            )
            .def_readonly("outMagneticField", reinterpret_cast<ProviderFor<LightH,Geometry3D> FourierSolver3D::Reflected::*>
                                              (&FourierSolver3D::Reflected::outMagneticField),
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightH", "3D", "magnetic field", "A/m", "", "", "", "outMagneticField").c_str()
            )
            .def_readonly("outLightMagnitude", reinterpret_cast<ProviderFor<LightMagnitude,Geometry3D> FourierSolver3D::Reflected::*>
                                              (&FourierSolver3D::Reflected::outLightMagnitude),
                format(docstring_attr_provider<FIELD_PROPERTY>(), "LightMagnitude", "3D", "light intensity", "W/m²", "", "", "", "outLightMagnitude").c_str()
            )
        ;

        FourierSolver3D_LongTranWrapper<size_t>::register_("Sizes");
        FourierSolver3D_LongTranWrapper<PML>::register_("PMLs");
        FourierSolver3D_SymmetryLongTranWrapper::register_();
    }
    
    {CLASS(BesselSolverCyl, "BesselCyl",
        "Optical Solver using Bessel expansion in cylindrical coordinates.\n\n"
        "It calculates optical modes and optical field distribution using Bessel slab method\n"
        "and reflection transfer in two-dimensional cylindrical space.")
        export_base(solver);
//         solver.add_property("material_mesh", &__Class__::getMesh, "Regular mesh with points in which material is sampled.");
        PROVIDER(outWavelength, "");
        PROVIDER(outLoss, "");
        METHOD(find_mode, findMode, 
               "Compute the mode near the specified effective index.\n\n"
               "Only one of the following arguments can be given through a keyword.\n"
               "It is the starting point for search of the specified parameter.\n\n"
               "Args:\n"
               "    lam (complex): Wavelength.\n"
               "    k0 (complex): Normalized frequency.\n"
               "    neff (complex): Longitudinal effective index.\n"
               "    ktran (complex): Transverse wavevector.\n",
               "lam", arg("m")=1
              );
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size.");
        solver.def("get_determinant", py::raw_function(BesselSolverCyl_getDeterminant),
                   "Compute discontinuity matrix determinant.\n\n"
                   "Arguments can be given through keywords only.\n\n"
                   "Args:\n"
                   "    lam (complex): Wavelength.\n"
                   "    k0 (complex): Normalized frequency.\n");
//         solver.def("compute_reflectivity", &FourierSolver_computeReflectivity<FourierSolver2D>,
//                    "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
//                    "Args:\n"
//                    "    lam (float or array of floats): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                    "        name of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    , (py::arg("lam"), "polarization", "side");
//         solver.def("compute_transmittivity", &FourierSolver_computeTransmittivity<FourierSolver2D>,
//                    "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
//                    "Args:\n"
//                    "    lam (float or array of floats): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                    "        of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    , (py::arg("lam"), "polarization", "side"));
//         solver.def("compute_reflected_orders", &FourierSolver2D_reflectedAmplitudes,
//                    "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
//                    "Args:\n"
//                    "    lam (float): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                    "        name of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    , (py::arg("lam"), "polarization", "side"));
//         solver.def("compute_transmitted_orders", &FourierSolver2D_transmittedAmplitudes,
//                    "Compute Fourier coefficients of the reflected field on the perpendicular incidence [-].\n\n"
//                    "Args:\n"
//                    "    lam (float): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                    "        name of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    , (py::arg("lam"), "polarization", "side"));
//         solver.add_property("mirrors", FourierSolver2D_getMirrors, FourierSolver2D_setMirrors,
//                    "Mirror reflectivities. If None then they are automatically estimated from the\n"
//                    "Fresnel equations.");
        solver.add_property("pml", py::make_function(&Solver_getPML<BesselSolverCyl>, py::with_custodian_and_ward_postcall<0,1>()),
                            &Solver_setPML<BesselSolverCyl>,
                            "Side Perfectly Matched Layers boundary conditions.\n\n"
                            PML_ATTRS_DOC
                           );
        RO_FIELD(modes, "Computed modes.");
//         solver.def("reflected", &FourierSolver_getReflected<FourierSolver2D>, py::with_custodian_and_ward_postcall<0,1>(),
//                    "Access to the reflected field.\n\n"
//                    "Args:\n"
//                    "    lam (float): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                    "        of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n\n"
//                    ":rtype: Fourier2D.Reflected\n"
//                    , (py::arg("lam"), "polarization", "side"));

#ifndef NDEBUG
        solver.add_property("wavelength", &SlabBase::getWavelength, &Solver_setWavelength<__Class__>, "Wavelength of the light [nm].");
        solver.add_property("k0", &__Class__::getK0, &Solver_setK0<__Class__>, "Normalized frequency of the light [1/µm].");
        solver.add_property("m", &__Class__::getM, &__Class__::setM, "Angular dependence parameter.");
        METHOD(ieps_minus, ieps_minus, "J_{m-1}(gr) eps^{-1}(r) J_{m-1}(kr) r dr", "layer");
        METHOD(ieps_plus, ieps_plus, "J_{m+1}(gr) eps^{-1}(r) J_{m+1}(kr) r dr", "layer");
        METHOD(eps_minus, eps_minus, "J_{m-1}(gr) eps(r) J_{m-1}(kr) r dr", "layer");
        METHOD(eps_plus, eps_plus, "J_{m+1}(gr) eps(r) J_{m+1}(kr) r dr", "layer");
        METHOD(deps_minus, deps_minus, "J_{m-1}(gr) deps/dr J_{m}(kr) r dr", "layer");
        METHOD(deps_plus, deps_plus, "_{m+1}(gr) deps/dr J_{m}(kr) r dr", "layer");
#endif

        py::scope scope = solver;

        register_vector_of<BesselSolverCyl::Mode>("Modes");
        py::class_<BesselSolverCyl::Mode>("Mode", "Detailed information about the mode.", py::no_init)
            .add_property("lam", &getModeWavelength<BesselSolverCyl::Mode>, "Mode wavelength [nm].")
            .add_property("wavelength", &getModeWavelength<BesselSolverCyl::Mode>, "Mode wavelength [nm].")
            .def_readonly("k0", &BesselSolverCyl::Mode::k0, "Mode normalized frequency [1/µm].")
            .def_readonly("m", &BesselSolverCyl::Mode::m, "Angular mode order.")
            .def_readwrite("power", &BesselSolverCyl::Mode::power, "Total power emitted into the mode.")
            .def("__str__", &BesselSolverCyl_Mode_str)
            .def("__repr__", &BesselSolverCyl_Mode_repr)
        ;
    }
    
}

