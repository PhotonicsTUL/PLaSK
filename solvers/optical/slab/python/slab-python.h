#ifndef PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.h>
#include <boost/python/raw_function.hpp>
using namespace plask;
using namespace plask::python;

#include "../matrices.h"
#include "../expansion.h"

#if defined(_WIN32) && defined(interface)
#   undef interface
#endif

namespace plask { namespace optical { namespace slab { namespace python {

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
    PythonDataVector<const dcomplex,2> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}


template <typename SolverT>
static py::object Solver_getInterface(SolverT& self) {
    size_t interface = self.getInterface();
    if (interface == size_t(-1)) return py::object();
    else return py::object(interface);
}

template <typename SolverT>
static void Solver_setInterface(SolverT& self, const py::object& value) {
    throw AttributeError("Setting interface by layer index is not supported anymore (set it by object or position)");
}


template <typename SolverT>
static py::object Solver_getLam0(const SolverT& self) {
    if (!isnan(self.lam0)) return py::object(self.lam0);
    else return py::object();
}

template <typename SolverT>
static void Solver_setLam0(SolverT& self, py::object value) {
    if (value == py::object()) self.clearLam0();
    else self.setLam0(py::extract<double>(value));
}

template <typename SolverT>
static py::tuple SlabSolver_getStack(SolverT& self) {
    self.initCalculation();
    py::list result;
    for (auto i: self.getStack()) {
        result.append(i);
    }
    return py::tuple(result);
}

template <typename SolverT>
static shared_ptr<OrderedAxis> SlabSolver_getLayerEdges(SolverT& self) {
    self.initCalculation();
    return make_shared<OrderedAxis>(*self.vbounds);
}

template <typename SolverT>
static shared_ptr<OrderedAxis> SlabSolver_getLayerCenters(SolverT& self) {
    self.initCalculation();
    return make_shared<OrderedAxis>(*self.verts);
}

// template <typename SolverT>
// static py::tuple SlabSolver_getLayerSets(const SolverT& self) {
//     py::list result;
//     for (auto i: self.getLayersPoints()) {
//         result.append(i);
//     }
//     return py::tuple(result);
// }

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
static void Solver_setK0(SolverT& self, dcomplex k0) { self.setK0(k0); }

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

template <typename Mode>
static double getModeLoss(const Mode& mode) {
    return 2e4 * mode.k0.imag();
}

#ifndef NDEBUG
template <typename Solver>
py::tuple Solver_getMatrices(Solver& self, size_t layer) {
    cmatrix RE, RH;
    self.getMatrices(layer, RE, RH);
    return py::make_tuple(py::object(RE), py::object(RH));
}

template <typename Solver>
py::tuple Solver_getDiagonalized(Solver& self, size_t layer) {
    self.initCalculation();
    if (!self.transfer) {
        self.initTransfer(self.getExpansion(), false);
        self.transfer->initDiagonalization();
        self.transfer->diagonalizer->diagonalizeLayer(layer);
    } else if (!self.transfer->diagonalizer->isDiagonalized(layer)) {
        self.transfer->diagonalizer->diagonalizeLayer(layer);
    }
    cdiagonal gamma = self.transfer->diagonalizer->Gamma(layer);
    cmatrix TE = self.transfer->diagonalizer->TE(layer),
            TH = self.transfer->diagonalizer->TH(layer);
    return py::make_tuple(py::object(gamma), py::object(TE), py::object(TH));
}
#endif

struct Eigenmodes {
    cdiagonal gamma;
    cmatrix TE, TH;

    template <typename Solver>
    Eigenmodes(Solver& solver, size_t layer) {
        bool changed = solver.initCalculation() || solver.setExpansionDefaults(true);
        if (!solver.transfer) {
            solver.initTransfer(solver.getExpansion(), false);
            changed = true;
        }
        if (changed) {
            solver.transfer->initDiagonalization();
            solver.transfer->diagonalizer->diagonalizeLayer(layer);
        } else if (!solver.transfer->diagonalizer->isDiagonalized(layer))
            solver.transfer->diagonalizer->diagonalizeLayer(layer);
        gamma = solver.transfer->diagonalizer->Gamma(layer);
        TE = solver.transfer->diagonalizer->TE(layer),
        TH = solver.transfer->diagonalizer->TH(layer);
    }

    size_t size() const {
        return gamma.size();
    }

  protected:
    virtual std::string getSolverId() const = 0;

    size_t index(int n) const {
        int N = gamma.size();
        if (n < 0) n += N;
        if (n < 0 || n >= N) throw IndexError("{}: Bad eigenmode number", getSolverId());
        return size_t(n);
    }

  public:
    dcomplex Gamma(int n) const {
        return gamma[index(n)];
    }
};


template <typename SolverT>
py::object Solver_computeReflectivity(SolverT* self,
                                      py::object wavelength,
                                      Expansion::Component polarization,
                                      Transfer::IncidentDirection incidence
                                     )
{
    if (!self->initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        self->expansion.setK0(2e3*M_PI/lam);
        return 100. * self->getReflection(polarization, incidence);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeTransmittivity(SolverT* self,
                                        py::object wavelength,
                                        Expansion::Component polarization,
                                        Transfer::IncidentDirection incidence
                                       )
{
    if (!self->initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        self->expansion.setK0(2e3*M_PI/lam);
        return 100. * self->getTransmission(polarization, incidence);
    }, wavelength);
}

template <typename SolverT>
py::object get_max_temp_diff(SolverT* self) {
    double value = self->getMaxTempDiff();
    if (isnan(value) || isinf(value)) return py::object();
    return py::object(value);
}

template <typename SolverT>
void set_max_temp_diff(SolverT* self, py::object value) {
    if (value == py::object()) self->setMaxTempDiff(NAN);
    else self->setMaxTempDiff(py::extract<double>(value));
}



template <typename Class>
inline void export_base(Class solver) {
    typedef typename Class::wrapped_type Solver;
    solver.add_property("interface", &Solver_getInterface<Solver>, &Solver_setInterface<Solver>, "Matching interface position.");
    solver.def("set_interface", (void(Solver::*)(const shared_ptr<const GeometryObject>&, const PathHints&))&Solver::setInterfaceOn,
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
    solver.add_property("layer_edges", &SlabSolver_getLayerEdges<Solver>, "Vertical posiotions of egges of each layer.");
    solver.add_property("layer_centers", &SlabSolver_getLayerCenters<Solver>,
                        "Vertical posiotions of centers of each layer.\n\n"
                        "At these positions materials and temperatures are probed.\n");
    // solver.add_property("layer_sets", &SlabSolver_getLayerSets<Solver>, "Vertical positions of layers in each layer set.");
    solver.add_property("group_layers", &Solver::getGroupLayers, &Solver::setGroupLayers,
                        "Layer grouping switch.\n\n"
                        "If this property is ``True``, similar layers are grouped for efficiency.");
    solver.add_property("temp_diff", &get_max_temp_diff<Solver>, &set_max_temp_diff<Solver>,
                        "Maximum temperature difference between the layers in one group.\n\n"
                        "If a temperature in a single layer varies vertically more than this value,\n"
                        "the layer is split into two and put into separate groups. If this is empty,\n"
                        "temperature gradient is ignored in layers grouping.\n\n");
    solver.add_property("temp_dist", &Solver::getTempDist, &Solver::setTempDist,
                        "Temperature probing step.\n\n"
                        "If :attr:`temp_diff` is not ``None``, the temperature is laterally probed\n"
                        "in points approximately separated by this distance.\n");
    solver.add_property("temp_layer", &Solver::getTempLayer, &Solver::setTempLayer,
                        "Temperature probing step.\n\n"
                        "If :attr:`temp_diff` is not ``None``, this is the minimum thickness of sublayers\n"
                        "resulting from temperature-gradient division.\n");
    solver.template add_receiver<ReceiverFor<Temperature, typename Solver::SpaceType>, Solver>("inTemperature", &Solver::inTemperature, "");
    solver.template add_receiver<ReceiverFor<Gain, typename Solver::SpaceType>, Solver>("inGain", &Solver::inGain, "");
    solver.add_provider("outRefractiveIndex", &Solver::outRefractiveIndex, "");
    solver.add_provider("outLightMagnitude", &Solver::outLightMagnitude, "");
    solver.add_provider("outLightE", &Solver::outLightE, "");
    solver.def_readonly("outElectricField", reinterpret_cast<ProviderFor<LightE, typename Solver::SpaceType> Solver::*>(&Solver::outLightE),
            "Alias for :attr:`outLightE`.");
    solver.add_provider("outLightH", &Solver::outLightH, "");
    solver.def_readonly("outMagneticField", reinterpret_cast<ProviderFor<LightH, typename Solver::SpaceType> Solver::*>(&Solver::outLightH),
            "Alias for :attr:`outLightH`.");
    solver.add_provider("outElectricField", &Solver::outLightE, "This is an alias for :attr:`outLightE`.");
    solver.add_provider("outMagneticField", &Solver::outLightH, "This is an alias for :attr:`outLightH`.");
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
    solver.add_property("lam0", Solver_getLam0<Solver>, Solver_setLam0<Solver>,
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
    solver.def("get_diagonalized", Solver_getDiagonalized<Solver>);
#endif
}


}}}} // # namespace plask::optical::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

