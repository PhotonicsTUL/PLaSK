#ifndef PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

#include <cmath>
#include <plask/python.hpp>
#include <util/ufunc.h>
#include <boost/python/raw_function.hpp>
using namespace plask;
using namespace plask::python;

#include "../matrices.h"
#include "../expansion.h"

#if defined(_WIN32) && defined(interface)
#   undef interface
#endif

namespace plask { namespace solvers { namespace slab { namespace python {

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


template <NPY_TYPES type>
static inline py::object arrayFromVec2D(cvector data, bool sep, int dim=1) {
    int strid = 2;
    if (sep) strid = dim = 1;
    npy_intp dims[] = { data.size() / strid, strid };
    npy_intp strides[] = { strid * sizeof(dcomplex), sizeof(dcomplex) };
    PyObject* arr = PyArray_New(&PyArray_Type, dim, dims, type, strides, (void*)data.data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from field coefficients");
    DataVectorWrap<const dcomplex,2> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}


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
        return plask::make_shared<PmlWrapper>(nullptr, new PML(factor, size, dist, order));
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
        return format("<factor: {0}, size: {1}, dist: {2}, shape: {3}>", str(pml->factor), pml->size, pml->dist, pml->order);
    }

    std::string __repr__() const {
        return format("PML(factor={0}, size={1}, dist={2}, shape={3})", str(pml->factor), pml->size, pml->dist, pml->order);
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
static void Solver_setWavelength(SolverT& self, dcomplex lam) { self.setWavelength(lam); }

template <typename SolverT>
static void Solver_setK0(SolverT& self, dcomplex k0) { self.setWavelength(k0); }

template <typename SolverT>
static PmlWrapper Solver_vPML(SolverT* self) {
    return PmlWrapper(self, &self->vpml);
}

template <typename SolverT>
static void Solver_setvPML(SolverT* self, const PmlWrapper& value) {
    self->vpml = *value.pml;
    self->invalidate();
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

template <typename Mode>
static dcomplex getModeWavelength(const Mode& mode) {
    return 2e3 * M_PI / mode.k0;
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


}}}} // # namespace plask::solvers::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

