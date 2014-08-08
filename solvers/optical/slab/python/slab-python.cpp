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
using namespace plask::solvers::slab;

#define ROOTDIGGER_ATTRS_DOC \
    ".. rubric:: Attributes\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.slab.RootParams.alpha\n" \
    "   ~optical.slab.RootParams.lambda\n" \
    "   ~optical.slab.RootParams.initial_range\n" \
    "   ~optical.slab.RootParams.maxiter\n" \
    "   ~optical.slab.RootParams.maxstep\n" \
    "   ~optical.slab.RootParams.method\n" \
    "   ~optical.slab.RootParams.tolf_max\n" \
    "   ~optical.slab.RootParams.tolf_min\n" \
    "   ~optical.slab.RootParams.tolx\n"

#define PML_ATTRS_DOC \
    ".. rubric:: Attributes\n\n" \
    ".. autosummary::\n\n" \
    "   ~optical.slab.PML.factor\n" \
    "   ~optical.slab.PML.shape\n" \
    "   ~optical.slab.PML.dist\n" \
    "   ~optical.slab.PML.size\n"

template <typename SolverT>
static const std::vector<std::size_t>& SlabSolver_getStack(const SolverT& self) { return self.getStack(); }

template <typename SolverT>
static const std::vector<OrderedAxis>& SlabSolver_getLayerSets(const SolverT& self) { return self.getLayersPoints(); }

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

    static shared_ptr<PmlWrapper> __init__(dcomplex factor, double size, double shift, double order) {
        return make_shared<PmlWrapper>(nullptr, new PML(factor, size, shift, order));
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

    double get_shift() const { return pml->shift; }
    void set_shift(double val) {
        pml->shift = val;
        if (solver) solver->invalidate();
    }

    double get_order() const { return pml->order; }
    void set_order(double val) {
        pml->order = val;
        if (solver) solver->invalidate();
    }

    std::string __str__() const {
        return format("<factor: %1%, size: %2%, dist: %3%, shape: %4%>", str(pml->factor), pml->size, pml->shift, pml->order);
    }

    std::string __repr__() const {
        return format("PML(factor=%1%, size=%2%, dist=%3%, shape=%4%)", str(pml->factor), pml->size, pml->shift, pml->order);
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


dcomplex FourierSolver2D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver2D* self = py::extract<FourierSolver2D*>(args[0]);

    AxisNames* axes = getCurrentAxes();
    boost::optional<dcomplex> lambda, neff, ktran;
    bool dispersive = true;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength")
            lambda.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "k0")
            lambda.reset(2e3*M_PI / dcomplex(py::extract<dcomplex>(kwargs[*i])));
        else if (*i == "neff")
            neff.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran())
            ktran.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "dispersive")
            dispersive = py::extract<bool>(kwargs[*i]);
        else
            throw TypeError("determinant() got unexpected keyword argument '%1%'", *i);
    }
    if (lambda) self->setWavelength(*lambda, dispersive);
    if (neff) self->setKlong(*neff * self->getK0());
    if (ktran) self->setKtran(*ktran);

    return self->getDeterminant();
}

py::object FourierSolver2D_computeReflectivity(FourierSolver2D* self,
                                                   py::object wavelength,
                                                   ExpansionPW2D::Component polarization,
                                                   Transfer::IncidentDirection incidence,
                                                   bool dispersive
                                                  )
{
    return UFUNC<double>([=](double lam)->double {
        self->setWavelength(lam, dispersive);
        return 100. * self->getReflection(polarization, incidence);
    }, wavelength);
}

py::object FourierSolver2D_computeTransmitticity(FourierSolver2D* self,
                                                     py::object wavelength,
                                                     ExpansionPW2D::Component polarization,
                                                     Transfer::IncidentDirection incidence,
                                                     bool dispersive
                                                    )
{
    return UFUNC<double>([=](double lam)->double {
        self->setWavelength(lam, dispersive);
        return 100. * self->getTransmission(polarization, incidence);
    }, wavelength);
}

PmlWrapper FourierSolver2D_PML(FourierSolver2D* self) {
    return PmlWrapper(self, &self->pml);
}

void FourierSolver2D_setPML(FourierSolver2D* self, const PmlWrapper& value) {
    self->pml = *value.pml;
    self->invalidate();
}

shared_ptr<FourierSolver2D::Reflected> FourierSolver2D_getReflected(FourierSolver2D* parent,
                                                                double wavelength,
                                                                ExpansionPW2D::Component polarization,
                                                                Transfer::IncidentDirection side)
{
    return make_shared<FourierSolver2D::Reflected>(parent, wavelength, polarization, side);
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
    return 2e3 / mode.k0;
}


py::object FourierSolver3D_Mode__getattr__(const FourierSolver3D::Mode& mode, const std::string name) {
    auto axes = getCurrentAxes();
    if (name == "k"+axes->getNameForLong()) return py::object(mode.klong);
    if (name == "k"+axes->getNameForTran()) return py::object(mode.ktran);
    throw AttributeError("'Mode' object has no attribute '%1%'", name);
    return py::object();
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
        try { if (py::len(values) != 2) throw py::error_already_set(); }
        catch (py::error_already_set) { throw TypeError("You may only assign a sequence of two values"); }
        self.setSymmetryLong(py::extract<Expansion::Component>(values[0]));
        self.setSymmetryTran(py::extract<Expansion::Component>(values[1]));
    }
};



dcomplex FourierSolver3D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    AxisNames* axes = getCurrentAxes();
    py::stl_input_iterator<std::string> begin(kwargs), end;
    bool dispersive = true;
    boost::optional<dcomplex> wavelength, k0;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam" || *i == "wavelength")
            wavelength.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "k0")
            k0.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "klong" || *i == "kl" || *i == "k"+axes->getNameForLong())
            self->setKlong(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "ktran" || *i == "kt" || *i == "k"+axes->getNameForTran())
            self->setKtran(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "dispersive")
            dispersive = py::extract<bool>(kwargs[*i]);
        else
            throw TypeError("determinant() got unexpected keyword argument '%1%'", *i);
    }

    if (wavelength) self->setWavelength(*wavelength, dispersive);
    if (k0) self->setK0(*k0, dispersive);

    return self->getDeterminant();
}

size_t FourierSolver3D_findMode(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierSolver3D* self = py::extract<FourierSolver3D*>(args[0]);

    if (py::len(kwargs) != 1)
        throw TypeError("determinant() takes exactly one keyword argument (%1% given)", py::len(kwargs));
    std::string key = py::extract<std::string>(kwargs.keys()[0]);
    dcomplex value = py::extract<dcomplex>(kwargs[key]);
    AxisNames* axes = getCurrentAxes();
    FourierSolver3D::What what;

    if (key == "lam" || key == "wavelength")
        what = FourierSolver3D::WHAT_WAVELENGTH;
    else if (key == "k0")
        what = FourierSolver3D::WHAT_K0;
    else if (key == "klong" || key == "kl" || key == "k"+axes->getNameForLong())
        what = FourierSolver3D::WHAT_KLONG;
    else if (key == "ktran" || key == "kt" || key == "k"+axes->getNameForTran())
        what = FourierSolver3D::WHAT_KTRAN;
    else
        throw TypeError("determinant() got unexpected keyword argument '%1%'", key);

    return self->findMode(what, value);
}



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
    solver.add_property("stack", py::make_function<>(&SlabSolver_getStack<Solver>, py::return_internal_reference<>()), "Stack of distinct layers.");
    solver.add_property("layer_sets", py::make_function<>(&SlabSolver_getLayerSets<Solver>, py::return_internal_reference<>()), "Vertical positions of layers in each layer set.");
    solver.add_receiver("inTemperature", &Solver::inTemperature, "");
    solver.add_receiver("inGain", &Solver::inGain, "");
    solver.add_provider("outRefractiveIndex", &Solver::outRefractiveIndex, "");
    solver.add_provider("outLightMagnitude", &Solver::outLightMagnitude, "");
    solver.add_provider("outElectricField", &Solver::outElectricField, "");
    solver.add_provider("outMagneticField", &Solver::outMagneticField, "");
    solver.add_property("wavelength", &Solver::getWavelength, &Solver_setWavelength<Solver>, "Wavelength of the light [nm].");
    solver.add_property("k0", &Solver::getK0, &Solver_setK0<Solver>, "Normalized frequency of the light [1/µm].");
    solver.add_property("klong", &Solver::getKlong, &Solver::setKlong, "Longitudinal propagation constant of the light [1/µm].");
    solver.add_property("ktran", &Solver::getKtran, &Solver::setKtran, "Transverse propagation constant of the light [1/µm].");
    solver.def_readwrite("root", &Solver::root,
                         "Configuration of the root searching algorithm for horizontal component of the\n"
                         "mode.\n\n"
                         ROOTDIGGER_ATTRS_DOC
                        );
}

BOOST_PYTHON_MODULE(slab)
{
    plask_import_array();

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
        .add_property("dist", &PmlWrapper::get_shift, &PmlWrapper::set_shift, "PML distance from the structure.")
        .add_property("shape", &PmlWrapper::get_order, &PmlWrapper::set_order, "PML shape order (0 → flat, 1 → linearly increasing, 2 → quadratic, etc.).")
        .def("__str__", &PmlWrapper::__str__)
        .def("__repr__", &PmlWrapper::__repr__)
    ;

    py_enum<RootDigger::Method>()
        .value("MULLER", RootDigger::ROOT_MULLER)
        .value("BROYDEN", RootDigger::ROOT_BROYDEN)
    ;

    py_enum<typename ReflectionTransfer::IncidentDirection>()
        .value("TOP", ReflectionTransfer::INCIDENCE_TOP)
        .value("BOTTOM", ReflectionTransfer::INCIDENCE_BOTTOM)
    ;

    py::class_<RootDigger::Params, boost::noncopyable>("RootParams", "Configuration of the root finding algorithm.", py::no_init)
        .def_readwrite("method", &RootDigger::Params::method, "Root finding method ('muller' or 'broyden')")
        .def_readwrite("tolx", &RootDigger::Params::tolx, "Absolute tolerance on the argument.")
        .def_readwrite("tolf_min", &RootDigger::Params::tolf_min, "Sufficient tolerance on the function value.")
        .def_readwrite("tolf_max", &RootDigger::Params::tolf_max, "Required tolerance on the function value.")
        .def_readwrite("maxiter", &RootDigger::Params::maxiter, "Maximum number of iterations.")
        .def_readwrite("maxstep", &RootDigger::Params::maxstep, "Maximum step in one iteration (Broyden method only).")
        .def_readwrite("alpha", &RootDigger::Params::maxstep, "Parameter ensuring sufficient decrease of determinant in each step\n(Broyden method only).")
        .def_readwrite("lambda", &RootDigger::Params::maxstep, "Minimum decrease ratio of one step (Broyden method only).")
        .def_readwrite("initial_range", &RootDigger::Params::initial_dist, "Initial range size (Muller method only).")
    ;

    {CLASS(FourierSolver2D, "Fourier2D",
        "Optical Solver using Fourier expansion in 2D.\n\n"
        "It calculates optical modes and optical field distribution using Fourier slab method\n"
        "and reflection transfer in two-dimensional Cartesian space.")
        export_base(solver);
        PROVIDER(outNeff, "Effective index of the last computed mode.");
        METHOD(find_mode, findMode,
               "Compute the mode near the specified effective index.\n\n"
               "Args:\n"
               "    neff (complex): Starting effective index.\n"
               , "neff");
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size.");
        RW_PROPERTY(symmetry, getSymmetry, setSymmetry, "Mode symmetry.");
        RW_PROPERTY(polarization, getPolarization, setPolarization, "Mode polarization.");
        RW_FIELD(refine, "Number of refinement points for refractive index averaging.");
        solver.def("determinant", py::raw_function(FourierSolver2D_getDeterminant),
                   "Compute discontinuity matrix determinant.\n\n"
                   "Arguments can be given through keywords only.\n\n"
                   "Args:\n"
                   "    lam (complex): Wavelength.\n"
                   "    k0 (complex): Normalized frequency.\n"
                   "    neff (complex): Effective index.\n"
                   "    ktran (complex): Transverse wavevector.\n"
                   "    dispersive (bool): If ``False`` then material coefficients are not\n"
                   "                       recomputed even if the wavelength is changed.\n");
        solver.def("compute_reflectivity", &FourierSolver2D_computeReflectivity,
                   "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
                   "Args:\n"
                   "    lam (float or array of floats): Incident light wavelength.\n"
                   "    polarization: Specification of the incident light polarization.\n"
                   "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
                   "        name of the non-vanishing electric field component.\n"
                   "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                   "        present.\n"
                   "    dispersive (bool): If *True*, material parameters will be recomputed at each\n"
                   "        wavelength, as they may change due to the dispersion.\n"
                   , (py::arg("lam"), "polarization", "side", py::arg("dispersive")=true));
        solver.def("compute_transmittivity", &FourierSolver2D_computeTransmitticity,
                   "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
                   "Args:\n"
                   "    lam (float or array of floats): Incident light wavelength.\n"
                   "    polarization: Specification of the incident light polarization.\n"
                   "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                   "        of the non-vanishing electric field component.\n"
                   "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                   "        present.\n"
                   "    dispersive (bool): If *True*, material parameters will be recomputed at each\n"
                   "        wavelength, as they may change due to the dispersion.\n"
                   , (py::arg("lam"), "polarization", "side", py::arg("dispersive")=true));
        solver.add_property("mirrors", FourierSolver2D_getMirrors, FourierSolver2D_setMirrors,
                   "Mirror reflectivities. If None then they are automatically estimated from the\n"
                   "Fresnel equations.");
        solver.add_property("pml", py::make_function(&FourierSolver2D_PML, py::with_custodian_and_ward_postcall<0,1>()),
                            &FourierSolver2D_setPML,
                            "Side Perfectly Matched Layers boundary conditions.\n\n"
                            PML_ATTRS_DOC
                           );
        RO_FIELD(modes, "Computed modes.");
        solver.def("reflected", &FourierSolver2D_getReflected, py::with_custodian_and_ward_postcall<0,1>(),
                   "Access to the reflected field.\n\n"
                   "Args:\n"
                   "    lam (float): Incident light wavelength.\n"
                   "    polarization: Specification of the incident light polarization.\n"
                   "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
                   "        of the non-vanishing electric field component.\n"
                   "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
                   "        present.\n"
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
        solver.def("determinant", py::raw_function(FourierSolver3D_getDeterminant),
                   "Compute discontinuity matrix determinant.\n\n"
                   "Arguments can be given through keywords only.\n\n"
                   "Args:\n"
                   "    lam (complex): Wavelength.\n"
                   "    k0 (complex): Normalized frequency.\n"
                   "    klong (complex): Longitudinal wavevector.\n"
                   "    ktran (complex): Transverse wavevector.\n"
                   "    dispersive (bool): If ``False`` then material coefficients are not\n"
                   "                       recomputed even if the wavelength is changed.\n");
        solver.def("find_mode", py::raw_function(FourierSolver3D_findMode),
                   "Compute the mode near the specified effective index.\n\n"
                   "Only one of the following arguments can be given through a keyword.\n"
                   "It is the starting point for search of the specified parameter.\n\n"
                   "Args:\n"
                   "    lam (complex): Wavelength.\n"
                   "    k0 (complex): Normalized frequency.\n"
                   "    klong (complex): Longitudinal wavevector.\n"
                   "    ktran (complex): Transverse wavevector.\n");
//         solver.def("compute_reflectivity", &FourierSolver2D_computeReflectivity,
//                    "Compute reflection coefficient on the perpendicular incidence [%].\n\n"
//                    "Args:\n"
//                    "    lam (float or array of floats): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis\n"
//                    "        name of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    "    dispersive (bool): If *True*, material parameters will be recomputed at each\n"
//                    "        wavelength, as they may change due to the dispersion.\n"
//                    , (py::arg("lam"), "polarization", "side", py::arg("dispersive")=true));
//         solver.def("compute_transmittivity", &FourierSolver2D_computeTransmitticity,
//                    "Compute transmission coefficient on the perpendicular incidence [%].\n\n"
//                    "Args:\n"
//                    "    lam (float or array of floats): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                    "        of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    "    dispersive (bool): If *True*, material parameters will be recomputed at each\n"
//                    "        wavelength, as they may change due to the dispersion.\n"
//                    , (py::arg("lam"), "polarization", "side", py::arg("dispersive")=true));
//         solver.def("reflected", &FourierSolver2D_getReflected, py::with_custodian_and_ward_postcall<0,1>(),
//                    "Access to the reflected field.\n\n"
//                    "Args:\n"
//                    "    lam (float): Incident light wavelength.\n"
//                    "    polarization: Specification of the incident light polarization.\n"
//                    "        It should be a string of the form 'E\\ *#*\\ ', where *#* is the axis name\n"
//                    "        of the non-vanishing electric field component.\n"
//                    "    side (`top` or `bottom`): Side of the structure where the incident light is\n"
//                    "        present.\n"
//                    , (py::arg("lam"), "polarization", "side"));
        RO_FIELD(modes, "Computed modes.");
        py::scope scope = solver;

        register_vector_of<FourierSolver3D::Mode>("Modes");
        py::class_<FourierSolver3D::Mode>("Mode", "Detailed information about the mode.", py::no_init)
//TODO             .def_readonly("symmetry", &FourierSolver3D::Mode::symmetry, "Mode horizontal symmetry.")
            .add_property("lam", &getModeWavelength<FourierSolver3D::Mode>, "Mode wavelength [nm].")
            .add_property("wavelength", &getModeWavelength<FourierSolver3D::Mode>, "Mode wavelength [nm].")
            .def_readonly("k0", &FourierSolver3D::Mode::k0, "Mode normalized frequency [1/µm].")
            .def_readonly("klong", &FourierSolver3D::Mode::klong, "Mode longitudinal wavevector [1/µm].")
            .def_readonly("ktran", &FourierSolver3D::Mode::ktran, "Mode transverse wavevector [1/µm].")
            .def_readwrite("power", &FourierSolver3D::Mode::power, "Total power emitted into the mode.")
            .def("__getattr__", &FourierSolver3D_Mode__getattr__)
        ;

        FourierSolver3D_LongTranWrapper<size_t>::register_("Sizes");
        FourierSolver3D_LongTranWrapper<PML>::register_("PMLs");
        FourierSolver3D_SymmetryLongTranWrapper::register_();
    }

}

