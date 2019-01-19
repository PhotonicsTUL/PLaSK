#ifndef PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H
#define PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

#include <cmath>
#include <cstring>
#include <plask/python.hpp>
#include <plask/python_numpy.h>
#include <plask/python_util/ufunc.h>
#include <boost/python/raw_function.hpp>
using namespace plask;
using namespace plask::python;

#include "../matrices.h"
#include "../expansion.h"
#include "../diagonalizer.h"

#include <plask/config.h>

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


template <NPY_TYPES type, typename T>
inline static py::object arrayFromVec(const DataVector<T>& data) {
    npy_intp dims[] = { npy_intp(data.size()) };
    npy_intp strides[] = { npy_intp(sizeof(T)) };
    PyObject* arr = PyArray_New(&PyArray_Type, 1, dims, type, strides, (void*)data.data(), 0, 0, NULL);
    if (arr == nullptr) throw plask::CriticalException(u8"Cannot create array from field coefficients");
    PythonDataVector<const T, 3> wrap(data);
    py::object odata(wrap); py::incref(odata.ptr());
    PyArray_SetBaseObject((PyArrayObject*)arr, odata.ptr()); // Make sure the data vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}


template <NPY_TYPES type>
static inline py::object arrayFromVec2D(cvector data, bool sep, int dim=1) {
    int strid = 2;
    if (sep) strid = dim = 1;
    npy_intp dims[] = { npy_intp(data.size() / strid), npy_intp(strid) };
    npy_intp strides[] = { npy_intp(strid * sizeof(dcomplex)), npy_intp(sizeof(dcomplex)) };
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
static void Solver_setInterface(SolverT& PLASK_UNUSED(self), const py::object& PLASK_UNUSED(value)) {
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
    self.Solver::initCalculation();
    py::list result;
    for (auto i: self.getStack()) {
        result.append(i);
    }
    return py::tuple(result);
}

template <typename SolverT>
static shared_ptr<OrderedAxis> SlabSolver_getLayerEdges(SolverT& self) {
    self.Solver::initCalculation();
    return make_shared<OrderedAxis>(*self.vbounds);
}

template <typename SolverT>
static shared_ptr<OrderedAxis> SlabSolver_getLayerCenters(SolverT& self) {
    self.Solver::initCalculation();
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

    // Determine if obj can be converted into component
    static void* convertible(PyObject* obj) {
        if (obj == Py_None) return obj;
        if (!PyString_Check(obj)) return nullptr;
        const char* repr = PyString_AsString(obj);
        if (!strcmp(repr, "none") || !strcmp(repr, "NONE") || !strcmp(repr, "None") ||
            ((repr[0] == 'E' || repr[0] == 'H') &&
             (repr[1] == 'l' || repr[1] == 't' || repr[1] == 'x' || repr[1] == 'y' || repr[1] == 'z' || repr[1] == 'r' || repr[1] == 'p')))
            return obj;
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
            } catch (py::error_already_set&) {
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
    template <typename S> static Wrapper make(S* /*solver*/, T* item) { return *item; }
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
    return 2e3 * PI / mode.k0;
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
    self.Solver::initCalculation();
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



/**
 * Proxy class for accessing layer eigenmodes
 */
template <typename SolverT>
struct Eigenmodes {
    cdiagonal gamma;
    cmatrix TE, TH;

    SolverT& solver;
    size_t layer;

    typename ProviderFor<ModeLightMagnitude, typename SolverT::SpaceType>::Delegate outLightMagnitude;
    typename ProviderFor<ModeLightE,typename SolverT::SpaceType>::Delegate outLightE;
    typename ProviderFor<ModeLightH,typename SolverT::SpaceType>::Delegate outLightH;


    Eigenmodes(SolverT& solver, size_t layer): solver(solver), layer(layer),
               outLightMagnitude(this, &Eigenmodes::getLightMagnitude, &Eigenmodes::size),
               outLightE(this, &Eigenmodes::getLightE, &Eigenmodes::size),
               outLightH(this, &Eigenmodes::getLightH, &Eigenmodes::size) {
        bool changed = solver.Solver::initCalculation() || solver.setExpansionDefaults(true);
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

    static shared_ptr<Eigenmodes<SolverT>> fromZ(SolverT& solver, double z) {
        solver.Solver::initCalculation();
        size_t layer = solver.stack[solver.getLayerFor(z)];
        return make_shared<Eigenmodes<SolverT>>(solver, layer);
    }

    size_t size() const {
        return gamma.size();
    }

    py::object array(const dcomplex* data, size_t N) const;

    LazyData<double> getLightMagnitude(std::size_t n, shared_ptr<const MeshD<SolverT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) {
        if (n >= gamma.size())
            throw IndexError("Bad eigenmode number");
        cvector E(TE.data() + TE.rows()*n, TE.rows());
        cvector H(TH.data() + TH.rows()*n, TH.rows());
        solver.transfer->diagonalizer->source()->initField(Expansion::FIELD_E, method);
        DataVector<double> destination(dst_mesh->size());
        auto levels = makeLevelsAdapter(dst_mesh);
        while (auto level = levels->yield()) {
            //TODO warn if z is outside of the layer
            double z = level->vpos();
            dcomplex phas = exp(- I * gamma[n] * z);
            //size_t n = solver->getLayerFor(z);
            auto dest = solver.transfer->diagonalizer->source()->getField(layer, level, E, H);
            for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = abs2(phas * dest[i]);
        }
        solver.transfer->diagonalizer->source()->cleanupField();
        return destination;
    }

    LazyData<Vec<3,dcomplex>> getLightE(std::size_t n, shared_ptr<const MeshD<SolverT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) {
        if (n >= gamma.size())
            throw IndexError("Bad eigenmode number");
        cvector E(TE.data() + TE.rows()*n, TE.rows());
        cvector H(TH.data() + TH.rows()*n, TH.rows());
        solver.transfer->diagonalizer->source()->initField(Expansion::FIELD_E, method);
        DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
        auto levels = makeLevelsAdapter(dst_mesh);
        while (auto level = levels->yield()) {
            //TODO warn if z is outside of the layer
            double z = level->vpos();
            dcomplex phas = exp(- I * gamma[n] * z);
            //size_t n = solver->getLayerFor(z);
            auto dest = solver.transfer->diagonalizer->source()->getField(layer, level, E, H);
            for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = phas * dest[i];
        }
        solver.transfer->diagonalizer->source()->cleanupField();
        return destination;
    }

    LazyData<Vec<3,dcomplex>> getLightH(std::size_t n, shared_ptr<const MeshD<SolverT::SpaceType::DIM>> dst_mesh, InterpolationMethod method) {
        if (n >= gamma.size())
            throw IndexError("Bad eigenmode number");
        cvector E(TE.data() + TE.rows()*n, TE.rows());
        cvector H(TH.data() + TH.rows()*n, TH.rows());
        solver.transfer->diagonalizer->source()->initField(Expansion::FIELD_H, method);
        DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
        auto levels = makeLevelsAdapter(dst_mesh);
        while (auto level = levels->yield()) {
            //TODO warn if z is outside of the layer
            double z = level->vpos();
            dcomplex phas = exp(- I * gamma[n] * z);
            //size_t n = solver->getLayerFor(z);
            auto dest = solver.transfer->diagonalizer->source()->getField(layer, level, E, H);
            for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = phas * dest[i];
        }
        solver.transfer->diagonalizer->source()->cleanupField();
        return destination;
    }

    struct Eigenmode {
        Eigenmodes* ems;
        size_t n;

        Eigenmode(Eigenmodes* eigenmodes, int n): ems(eigenmodes), n(n) {}

        dcomplex getGamma() const {
            return ems->gamma[n];
        }

        py::object getCoefficientsE() const {
            return ems->array(ems->TE.data() + ems->TE.rows()*n, ems->TE.rows());
        }

        py::object getCoefficientsH() const {
            return ems->array(ems->TH.data() + ems->TH.rows()*n, ems->TH.rows());
        }

        double getFlux() const {
            return abs(ems->solver.getExpansion().getModeFlux(n, ems->TE, ems->TH));
        }
    };

    Eigenmode __getitem__(int n) {
        if (n < 0) n = size() + n;
        if (n < 0 || n >= size())
            throw IndexError("Bad eigenmode number");
        return Eigenmode(this, n);
    }

    static void registerClass(const char* suffix) {
        py::class_<Eigenmodes, shared_ptr<Eigenmodes>, boost::noncopyable> ems("Eigenmodes",
            u8"Layer eignemodes\n\n"
            u8"This is an advanced class allowing to extract eignemodes in each layer.\n", py::no_init); ems
            .def("__len__", &Eigenmodes::size)
            .def("__getitem__", &Eigenmodes::__getitem__, py::with_custodian_and_ward_postcall<0,1>())
            .def_readonly("outLightMagnitude",
                          reinterpret_cast<ProviderFor<ModeLightMagnitude, typename SolverT::SpaceType> Eigenmodes::*> (&Eigenmodes::outLightMagnitude),
                          format(docstring_attr_provider<ModeLightMagnitude>(), "LightMagnitude", suffix, u8"light intensity", u8"W/m²", "", "", "", "outLightMagnitude", "n=0", ":param int n: Number of the mode found with :meth:`find_mode`.").c_str()
                         )
            .def_readonly("outLightE",
                          reinterpret_cast<ProviderFor<ModeLightE, typename SolverT::SpaceType> Eigenmodes::*> (&Eigenmodes::outLightE),
                          format(docstring_attr_provider<ModeLightE>(), "LightE", suffix, u8"electric field", u8"V/m", "", "", "", "outLightE", "n=0", ":param int n: Number of the mode found with :meth:`find_mode`.").c_str()
                         )
            .def_readonly("outLightH",
                          reinterpret_cast<ProviderFor<ModeLightE, typename SolverT::SpaceType> Eigenmodes::*> (&Eigenmodes::outLightH),
                          format(docstring_attr_provider<ModeLightE>(), "LightH", suffix, u8"electric field", u8"A/m", "", "", "", "outLightH", "n=0", ":param int n: Number of the mode found with :meth:`find_mode`.").c_str()
                         )
        ;
        py::scope scope(ems);
        py::class_<Eigenmode>("Eigenmode", py::no_init)
            .add_property("kvert", &Eigenmode::getGamma,
                 u8"Vertical propagation constant for the eigenmode.")
            .add_property("raw_E", &Eigenmode::getCoefficientsE,
                 u8"Electric field coefficients for the eigenmode.")
            .add_property("raw_H", &Eigenmode::getCoefficientsH,
                 u8"Magnetic field coefficients for the eigenmode.")
            .add_property("flux", &Eigenmode::getFlux,
                 u8"Vertical flux for the eigenmode.\n\n"
                 u8"This is equal to the vertical component of the Pointing vector integrated over\n"
                 u8"the numerical domain.\n"
             )
        ;
    }
};


struct CoeffsArray {

    PyArrayObject* array;

    CoeffsArray(PyArrayObject* arr): array(arr) {
        Py_XINCREF(array);
    }

    CoeffsArray(const CoeffsArray& src): array(src.array) {
        Py_XINCREF(array);
    }

    CoeffsArray(CoeffsArray&& src): array(src.array) {
        src.array = nullptr;
    }

    ~CoeffsArray() {
        Py_XDECREF(array);
    }

    static void* convertible(PyObject* obj);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);
};


/**
 * Proxy class for accessing scatterd fields
 */
template <typename SolverT>
struct Scattering {

    SolverT* solver;

    cvector incident;
    Transfer::IncidentDirection side;

    /// Provider of the optical electric field
    typename ProviderFor<LightE, typename SolverT::SpaceType>::Delegate outLightE;

    /// Provider of the optical magnetic field
    typename ProviderFor<LightH, typename SolverT::SpaceType>::Delegate outLightH;

    /// Provider of the optical field intensity
    typename ProviderFor<LightMagnitude, typename SolverT::SpaceType>::Delegate outLightMagnitude;


    double reflectivity() {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return solver->getReflection(incident, side);
    }

    double transmittivity() {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return solver->getTransmission(incident, side);
    }

    double reflectivity100() {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return 100. * solver->getReflection(incident, side);
    }

    double transmittivity100() {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return 100. * solver->getTransmission(incident, side);
    }

    struct Reflected {
        Scattering* parent;
        Reflected(Scattering* parent): parent(parent) {}
        py::object get_coefficients() {
            if (!parent->solver->initCalculation()) parent->solver->setExpansionDefaults();
            return arrayFromVec<NPY_CDOUBLE>(parent->solver->getReflectedCoefficients(parent->incident, parent->side));
        }
        py::object get_fluxes() {
            if (!parent->solver->initCalculation()) parent->solver->setExpansionDefaults();
            return arrayFromVec<NPY_DOUBLE>(parent->solver->getReflectedFluxes(parent->incident, parent->side));
        }
        shared_ptr<Eigenmodes<SolverT>> eigenmodes() {
            parent->solver->Solver::initCalculation();
            size_t layer = (parent->side == Transfer::INCIDENCE_BOTTOM)? 0 : parent->solver->stack.back();
            return make_shared<Eigenmodes<SolverT>>(*parent->solver, layer);
        }
    };
    Reflected get_reflected() { return Reflected(this); }

    struct Incident {
        Scattering* parent;
        Incident(Scattering* parent): parent(parent) {}
        py::object get_coefficients() {
            return arrayFromVec<NPY_CDOUBLE>(parent->incident);
        }
        py::object get_fluxes() {
            if (!parent->solver->initCalculation()) parent->solver->setExpansionDefaults();
            return arrayFromVec<NPY_DOUBLE>(parent->solver->getIncidentFluxes(parent->incident, parent->side));
        }
        shared_ptr<Eigenmodes<SolverT>> eigenmodes() {
            parent->solver->Solver::initCalculation();
            size_t layer = (parent->side == Transfer::INCIDENCE_BOTTOM)? 0 : parent->solver->stack.back();
            return make_shared<Eigenmodes<SolverT>>(*parent->solver, layer);
        }
    };
    Incident get_incident() { return Incident(this); }

    struct Transmitted {
        Scattering* parent;
        Transmitted(Scattering* parent): parent(parent) {}
        py::object get_coefficients() {
            if (!parent->solver->initCalculation()) parent->solver->setExpansionDefaults();
            return arrayFromVec<NPY_CDOUBLE>(parent->solver->getTransmittedCoefficients(parent->incident, parent->side));
        }
        py::object get_fluxes() {
            if (!parent->solver->initCalculation()) parent->solver->setExpansionDefaults();
            return arrayFromVec<NPY_DOUBLE>(parent->solver->getTransmittedFluxes(parent->incident, parent->side));
        }
        shared_ptr<Eigenmodes<SolverT>> eigenmodes() {
            parent->solver->Solver::initCalculation();
            size_t layer = (parent->side == Transfer::INCIDENCE_TOP)? 0 : parent->solver->stack.back();
            return make_shared<Eigenmodes<SolverT>>(*parent->solver, layer);
        }
    };
    Transmitted get_transmitted() { return Transmitted(this); }


    LazyData<Vec<3,dcomplex>> getLightE(const shared_ptr<const MeshD<SolverT::SpaceType::DIM>>& dst_mesh, InterpolationMethod method) {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return solver->getScatteredFieldE(incident, side, dst_mesh, method);
    }

    LazyData<Vec<3,dcomplex>> getLightH(const shared_ptr<const MeshD<SolverT::SpaceType::DIM>>& dst_mesh, InterpolationMethod method) {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return solver->getScatteredFieldH(incident, side, dst_mesh, method);
    }

    LazyData<double> getLightMagnitude(const shared_ptr<const MeshD<SolverT::SpaceType::DIM>>& dst_mesh, InterpolationMethod method) {
        if (!solver->initCalculation()) solver->setExpansionDefaults();
        return solver->getScatteredFieldMagnitude(incident, side, dst_mesh, method);
    }

    /**
     * Construct proxy.
     * \param wavelength incident light wavelength
     * \param side incidence sideif (incident.size() >= self->transfer->diagonalizer->matrixSize())
        throw BadInput(self->getId(), "Wrong incident vector size ({}, should be {}", incident.size(), solver->transfer->diagonalizer->matrixSize());

     * \param incident incident vector
     */
    Scattering(SolverT* solver, Transfer::IncidentDirection side, const cvector& incident):
        solver(solver), incident(incident), side(side),
        outLightE(this, &Scattering::getLightE),
        outLightH(this, &Scattering::getLightH),
        outLightMagnitude(this, &Scattering::getLightMagnitude) {
        solver->initCalculation();
        if (!solver->transfer) solver->initTransfer(solver->getExpansion(), true);
        if (incident.size() != solver->transfer->diagonalizer->matrixSize())
            throw BadInput(solver->getId(), "Wrong incident vector size ({}, should be {}", incident.size(), solver->transfer->diagonalizer->matrixSize());
    }

    /**
     * Construct proxy.
     * \param wavelength incident light wavelength
     * \param side incidence side
     * \param polarization polarization of the perpendicularly incident light
     */
    Scattering(SolverT* solver, Transfer::IncidentDirection side, Expansion::Component polarization):
        solver(solver), incident(solver->incidentVector(side, polarization)), side(side),
        outLightE(this, &Scattering::getLightE),
        outLightH(this, &Scattering::getLightH),
        outLightMagnitude(this, &Scattering::getLightMagnitude) {
    }

    /**
     * Construct proxy.
     * \param wavelength incident light wavelength
     * \param side incidence side
     * \param idx incident eigenmode index
     */
    Scattering(SolverT* solver, Transfer::IncidentDirection side, size_t idx):
        solver(solver), incident(solver->SlabBase::incidentVector(idx)), side(side),
        outLightE(this, &Scattering::getLightE),
        outLightH(this, &Scattering::getLightH),
        outLightMagnitude(this, &Scattering::getLightMagnitude) {
    }

    static shared_ptr<Scattering<SolverT>> from_polarization(SolverT* parent,
                                                             Transfer::IncidentDirection side,
                                                             Expansion::Component polarization) {
        return make_shared<Scattering<SolverT>>(parent, side, polarization);
    }

    static shared_ptr<Scattering<SolverT>> from_index(SolverT* parent,
                                                      Transfer::IncidentDirection side,
                                                      size_t idx) {
        return make_shared<Scattering<SolverT>>(parent, side, idx);
    }

    static shared_ptr<Scattering<SolverT>> from_array(SolverT* parent,
                                                      Transfer::IncidentDirection side,
                                                      CoeffsArray coeffs) {
        PyArrayObject* arr = coeffs.array;
        cvector incident((dcomplex*)PyArray_DATA(arr), size_t(PyArray_DIMS(arr)[0]), plask::python::detail::NumpyDataDeleter(arr));
        return shared_ptr<Scattering<SolverT>>(new Scattering<SolverT>(parent, side, incident));
    }

    static py::class_<Scattering<SolverT>, shared_ptr<Scattering<SolverT>>, boost::noncopyable> registerClass(const char* suffix, const char* name="Fourier") {
        py::class_<Scattering<SolverT>, shared_ptr<Scattering<SolverT>>, boost::noncopyable> cls("Scattering",
            u8"Reflected mode proxy.\n\n"
            u8"This class contains providers for the scattered field.\n"
            , py::no_init); cls
            .def_readonly("outLightE", reinterpret_cast<ProviderFor<LightE, typename SolverT::SpaceType> Scattering<SolverT>::*>
                                                (&Scattering<SolverT>::outLightE),
                format(docstring_attr_provider<LightE>(), "LightE", suffix, u8"electric field", u8"V/m", "", "", "", "outLightE").c_str() )
            .def_readonly("outLightH", reinterpret_cast<ProviderFor<LightH, typename SolverT::SpaceType> Scattering<SolverT>::*>
                                                (&Scattering<SolverT>::outLightH),
                format(docstring_attr_provider<LightH>(), "LightH", suffix, u8"magnetic field", u8"A/m", "", "", "", "outLightH").c_str() )
            .def_readonly("outLightMagnitude", reinterpret_cast<ProviderFor<LightMagnitude, typename SolverT::SpaceType> Scattering<SolverT>::*>
                                                (&Scattering<SolverT>::outLightMagnitude),
                format(docstring_attr_provider<LightMagnitude>(), "LightMagnitude", suffix, u8"light intensity", u8"W/m²", "", "", "", "outLightMagnitude").c_str() )

            .add_property("R", &Scattering<SolverT>::reflectivity, u8"Total reflection coefficient [-].")
            .add_property("T", &Scattering<SolverT>::transmittivity, u8"Total transmission coefficient [-].")
            .add_property("reflectivity", &Scattering<SolverT>::reflectivity100, u8"Total reflection coefficient [%].\n\nThis differs from :attr:`Scattering.R` by unit.\n")
            .add_property("transmittivity", &Scattering<SolverT>::transmittivity100, u8"Total transmission coefficient [%].\n\nThis differs from :attr:`Scattering.T` by unit.\n")

            .add_property("reflected", py::make_function(&Scattering<SolverT>::get_reflected, py::with_custodian_and_ward_postcall<0,1>()),
                          u8"Reflected field details.\n\n"
                          u8":rtype: Reflected")
            .add_property("transmitted", py::make_function(&Scattering<SolverT>::get_transmitted, py::with_custodian_and_ward_postcall<0,1>()),
                          u8"Transmitted field details.\n\n"
                          u8":rtype: Transmitted")
            .add_property("incident", py::make_function(&Scattering<SolverT>::get_incident, py::with_custodian_and_ward_postcall<0,1>()),
                          u8"Incident field details.\n\n"
                          u8":rtype: Incident")
        ;

        py::scope scope(cls);

        py::class_<Scattering<SolverT>::Reflected>("Reflected", "Reflected field details", py::no_init)
            .add_property("coeffs", &Scattering<SolverT>::Reflected::get_coefficients, "Raw reflection ceofficients for modes.")
            .add_property("fluxes", &Scattering<SolverT>::Reflected::get_fluxes, "Perpendicular fluxes for reflected modes.")
            .add_property("eigenmodes", py::make_function(&Scattering<SolverT>::Reflected::eigenmodes, py::with_custodian_and_ward_postcall<0,1>()),
                          format("Reflected eigenmodes.\n\n"
                                 ":rtype: :class:`~optical.slab.{}{}.Eigenmodes`", name, suffix).c_str()
                         )
        ;

        py::class_<Scattering<SolverT>::Transmitted>("Transmitted", "Transmitted field details", py::no_init)
            .add_property("coeffs", &Scattering<SolverT>::Transmitted::get_coefficients, "Raw transmission ceofficients for modes.")
            .add_property("fluxes", &Scattering<SolverT>::Transmitted::get_fluxes, "Perpendicular fluxes for transmitted modes.")
            .add_property("eigenmodes", py::make_function(&Scattering<SolverT>::Transmitted::eigenmodes, py::with_custodian_and_ward_postcall<0,1>()),
                          format("Transmitted eigenmodes.\n\n"
                                 ":rtype: :class:`~optical.slab.{}{}.Eigenmodes`", name, suffix).c_str()
                         )
        ;

        py::class_<Scattering<SolverT>::Incident>("Incident", "Incident field details", py::no_init)
            .add_property("coeffs", &Scattering<SolverT>::Incident::get_coefficients, "Raw incident ceofficients for modes.")
            .add_property("fluxes", &Scattering<SolverT>::Incident::get_fluxes, "Perpendicular fluxes for incident modes.")
            .add_property("eigenmodes", py::make_function(&Scattering<SolverT>::Incident::eigenmodes, py::with_custodian_and_ward_postcall<0,1>()),
                          format("Incident eigenmodes.\n\n"
                                 ":rtype: :class:`~optical.slab.{}{}.Eigenmodes`", name, suffix).c_str()
                         )
        ;

//             .def("get_electric_coefficients", FourierSolver2D_getReflectedFieldVectorE, py::arg("level"),
//                 u8"Get Fourier expansion coefficients for the electric field.\n\n"
//                 u8"This is a low-level function returning :math:`E_l` and/or :math:`E_t` Fourier\n"
//                 u8"expansion coefficients. Please refer to the detailed solver description for their\n"
//                 u8"interpretation.\n\n"
//                 u8"Args:\n"
//                 u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
//                 u8":rtype: numproxypy.ndarray\n"
//                 )
//             .def("get_magnetic_coefficients", FourierSolver2D_getReflectedFieldVectorH, py::arg("level"),
//                 u8"Get Fourier expansion coefficients for the magnegtic field.\n\n"
//                 u8"This is a low-level function returning :math:`H_l` and/or :math:`H_t` Fourier\n"
//                 u8"expansion coefficients. Please refer to the detailed solver description for their\n"
//                 u8"interpretation.\n\n"
//                 u8"Args:\n"
//                 u8"    level (float): Vertical level at which the coefficients are computed.\n\n"
//                 u8":rtype: numpy.ndarray\n"
//                 )
//         ;
        return cls;
    }
};




template <typename SolverT>
py::object Solver_computeReflectivity_polarization(SolverT* self,
                                                   py::object wavelength,
                                                   Transfer::IncidentDirection side,
                                                   Expansion::Component polarization
                                                  )
{
    if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->incidentVector(side, polarization, lam);
        self->expansion.setK0(k0);
        return 100. * self->getReflection(incident, side);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeTransmittivity_polarization(SolverT* self,
                                                     py::object wavelength,
                                                     Transfer::IncidentDirection side,
                                                     Expansion::Component polarization
                                                    )
{
    if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->incidentVector(side, polarization, lam);
        self->expansion.setK0(k0);
        return 100. * self->getTransmission(incident, side);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeReflectivity_index(SolverT* self,
                                            py::object wavelength,
                                            Transfer::IncidentDirection side,
                                            size_t index
                                           )
{
    if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->SlabBase::incidentVector(index);
        self->expansion.setK0(k0);
        return 100. * self->getReflection(incident, side);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeTransmittivity_index(SolverT* self,
                                              py::object wavelength,
                                              Transfer::IncidentDirection side,
                                              size_t index
                                             )
{
    if (!self->Solver::initCalculation())
        self->setExpansionDefaults(false);
    return UFUNC<double>([=](double lam)->double {
        double k0 = 2e3*PI/lam;
        cvector incident = self->SlabBase::incidentVector(index);
        self->expansion.setK0(k0);
        return 100. * self->getTransmission(incident, side);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeReflectivity_array(SolverT* self,
                                            py::object wavelength,
                                            Transfer::IncidentDirection side,
                                            CoeffsArray coeffs
                                           )
{
    if (!self->Solver::initCalculation()) self->setExpansionDefaults(false);
    if (!self->transfer) self->initTransfer(self->getExpansion(), true);

    PyArrayObject* arr = coeffs.array;
    size_t size(PyArray_DIMS(arr)[0]);
    if (size != self->transfer->diagonalizer->matrixSize())
        throw BadInput(self->getId(), "Wrong incident vector size ({}, should be {}", size, self->transfer->diagonalizer->matrixSize());

    cvector incident((dcomplex*)PyArray_DATA(arr), size_t(PyArray_DIMS(arr)[0]), plask::python::detail::NumpyDataDeleter(arr));

    return UFUNC<double>([self, incident, side](double lam)->double {
        double k0 = 2e3*PI/lam;
        self->expansion.setK0(k0);
        return 100. * self->getReflection(incident, side);
    }, wavelength);
}

template <typename SolverT>
py::object Solver_computeTransmittivity_array(SolverT* self,
                                              py::object wavelength,
                                              Transfer::IncidentDirection side,
                                              CoeffsArray coeffs
                                             )
{
    if (!self->Solver::initCalculation()) self->setExpansionDefaults(false);
    if (!self->transfer) self->initTransfer(self->getExpansion(), true);

    PyArrayObject* arr = coeffs.array;
    size_t size(PyArray_DIMS(arr)[0]);
    if (size != self->transfer->diagonalizer->matrixSize())
        throw BadInput(self->getId(), "Wrong incident vector size ({}, should be {}", size, self->transfer->diagonalizer->matrixSize());

    cvector incident((dcomplex*)PyArray_DATA(arr), size_t(PyArray_DIMS(arr)[0]), plask::python::detail::NumpyDataDeleter(arr));

    return UFUNC<double>([self, incident, side](double lam)->double {
        double k0 = 2e3*PI/lam;
        self->expansion.setK0(k0);
        return 100. * self->getTransmission(incident, side);
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


template <typename SolverT>
static double getIntegralEE(SolverT& self, int num, double z1, double z2) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return self.getIntegralEE(num, z1, z2);
}

template <typename SolverT>
static double getIntegralHH(SolverT& self, int num, double z1, double z2) {
    if (num < 0) num += int(self.modes.size());
    if (std::size_t(num) >= self.modes.size()) throw IndexError(u8"Bad mode number {:d}", num);
    return self.getIntegralHH(num, z1, z2);
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
    solver.add_provider("outLightH", &Solver::outLightH, "");
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
    solver.def("integrateEE", &getIntegralEE<Solver>, (py::arg("num"), "z1", "z2"),
               u8"Get average integral of the squared electric field:\n\n"
               u8"\\\\[\\\\frac 1 2 \\\\int_{z_1}^{z_2} \\|E\\|^2.\\\\]\n\n"
               u8"In the lateral direction integration is performed over the whole domain.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    z1 (float): Lower vertical bound of the integral.\n"
               u8"    z2 (float): Upper vertical bound of the integral.\n\n"
               u8"Returns:\n"
               u8"    float: Computed integral [V\\ :sup:`2` / m\\ :sup:`2`].\n"
              );
    solver.def("integrateHH", &getIntegralHH<Solver>, (py::arg("num"), "z1", "z2"),
               u8"Get average integral of the squared magnetic field:\n\n"
               u8"\\\\[\\\\frac 1 2 \\\\int_{z_1}^{z_2} \\|H\\|^2.\\\\]\n\n"
               u8"In the lateral direction integration is performed over the whole domain.\n\n"
               u8"Args:\n"
               u8"    num (int): Computed mode number.\n"
               u8"    z1 (float): Lower vertical bound of the integral.\n"
               u8"    z2 (float): Upper vertical bound of the integral.\n"
               u8"Returns:\n"
               u8"    float: Computed integral [A\\ :sup:`2` / m\\ :sup:`2`].\n"
              );

#ifndef NDEBUG
    solver.def("get_matrices", Solver_getMatrices<Solver>);
    solver.def("get_diagonalized", Solver_getDiagonalized<Solver>);
#endif
}


}}}} // # namespace plask::optical::slab::python

#endif // PLASK__SOLVER__OPTICAL__SLAB_PYTHON_H

