/** \file
 * Sample Python wrapper for your solver.
 */
#include <cmath>
#include <plask/python.hpp>
#include <boost/python/raw_function.hpp>
using namespace plask;
using namespace plask::python;

#if defined(_WIN32) && defined(interface)
#   undef interface
#endif

#include "../reflection_solver_2d.h"
#include "../reflection_solver_cyl.h"
using namespace plask::solvers::slab;

template <typename SolverT>
static const std::vector<std::size_t>& SlabSolver_getStack(const SolverT& self) { return self.getStack(); }

template <typename SolverT>
static const std::vector<RectilinearAxis>& SlabSolver_getLayerSets(const SolverT& self) { return self.getLayersPoints(); }

template <typename SolverT>
struct PmlWrapper {
    
    SolverT* solver;
    typename SolverT::PML* pml;
    
    PmlWrapper(SolverT* solver, typename SolverT::PML* pml): solver(solver), pml(pml) {}
    
    double get_extinction() const { return pml->extinction; }
    void set_extinction(double val) {
        pml->extinction = val;
        solver->invalidate();
    }
    
    double get_size() const { return pml->size; }
    void set_size(double val) {
        pml->size = val;
        solver->invalidate();
    }
    
    double get_shift() const { return pml->shift; }
    void set_shift(double val) {
        pml->shift = val;
        solver->invalidate();
    }
    
    double get_order() const { return pml->order; }
    void set_order(double val) {
        pml->order = val;
        solver->invalidate();
    }
    
    std::string __str__() const {
        return format("<extinction: %1%, size: %2%, shift: %3%, order: %4%>", pml->extinction, pml->size, pml->shift, pml->order);
    }
    
    std::string __repr__() const {
        return format("PML(extinction=%1%, size=%2%, shift=%3%, order=%4%)", pml->extinction, pml->size, pml->shift, pml->order);
    }
};

template <typename Class>
inline void export_base(Class solver) {
    typedef typename Class::wrapped_type Solver;
    solver.def_readwrite("outdist", &Solver::outdist, "Distance outside outer borders where material is sampled");
    solver.add_property("interface", &Solver::getInterface, &Solver::setInterface, "Matching interface position");
    solver.def("set_interface", (void(Solver::*)(const shared_ptr<GeometryObject>&, const PathHints&))&Solver::setInterfaceOn,
               "Set interface at the bottom of the object pointed by path", (py::arg("object"), py::arg("path")=py::object()));
    solver.def("set_interface", &Solver::setInterfaceAt, "Set interface around position pos", py::arg("pos"));
    solver.def_readwrite("smooth", &Solver::smooth, "Smoothing parameter");
    solver.add_property("stack", py::make_function<>(&SlabSolver_getStack<Solver>, py::return_internal_reference<>()), "Stack of distinct layers");
    solver.add_property("layer_sets", py::make_function<>(&SlabSolver_getLayerSets<Solver>, py::return_internal_reference<>()), "Vertical positions of layers in each layer set");
    solver.add_receiver("inTemperature", &Solver::inTemperature, "Optical gain in the active region");
    solver.add_receiver("inGain", &Solver::inGain, "Optical gain in the active region");
    solver.add_provider("outLightIntensity", &Solver::outLightIntensity, "Light intensity of the last computed mode");

    py::scope scope = solver;

    py::class_<PmlWrapper<Solver>>("PML", "Perfectly matched layer details", py::no_init)
        .add_property("extinction", &PmlWrapper<Solver>::get_extinction, &PmlWrapper<Solver>::set_extinction, "PML extinction parameter")
        .add_property("size", &PmlWrapper<Solver>::get_size, &PmlWrapper<Solver>::set_size, "PML size")
        .add_property("shift", &PmlWrapper<Solver>::get_shift, &PmlWrapper<Solver>::set_shift, "PML shift from the structure")
        .add_property("order", &PmlWrapper<Solver>::get_order, &PmlWrapper<Solver>::set_order, "PML shape order")
        .def("__str__", &PmlWrapper<Solver>::__str__)
        .def("__repr__", &PmlWrapper<Solver>::__repr__)
    ;
}

template <typename Class>
inline void export_reflection_base(Class solver) {
    export_base(solver);
    typedef typename Class::wrapped_type Solver;
    py::scope scope = solver;
    py_enum<typename Solver::IncidentDirection>("Incindent", "Direction of incident light for reflection calculations.")
        .value("TOP", Solver::DIRECTION_DOWNWARDS)
        .value("BOTTOM", Solver::DIRECTION_UPWARDS)
    ;
}



static py::object FourierReflection2D_getSymmetry(const FourierReflection2D& self) {
    AxisNames* axes = getCurrentAxes();
    switch (self.getSymmetry()) {
        case ExpansionPW2D::SYMMETRIC_E_TRAN: return py::object("E"+axes->getNameForTran());
        case ExpansionPW2D::SYMMETRIC_E_LONG: return py::object("E"+axes->getNameForLong());
        default: return py::object();
    }
}

static void FourierReflection2D_setSymmetry(FourierReflection2D& self, py::object symmetry) {
    AxisNames* axes = getCurrentAxes();
    if (symmetry == py::object()) { self.setSymmetry(ExpansionPW2D::SYMMETRIC_UNSPECIFIED); }
    try {
        std::string sym = py::extract<std::string>(symmetry);
        if (sym == "Etran" || sym == "E"+axes->getNameForTran())
            self.setSymmetry(ExpansionPW2D::SYMMETRIC_E_TRAN);
        else if (sym == "Elong" || sym == "E"+axes->getNameForLong())
            self.setSymmetry(ExpansionPW2D::SYMMETRIC_E_LONG);
        else throw py::error_already_set();
    } catch (py::error_already_set) {
        throw ValueError("Wrong symmetry specification.");
    }
}

void FourierReflection2D_parseKeywords(const char* name, FourierReflection2D* self, const py::dict& kwargs) {
    AxisNames* axes = getCurrentAxes();
    boost::optional<dcomplex> lambda, neff, ktran;
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "lam") lambda.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "neff") neff.reset(py::extract<dcomplex>(kwargs[*i]));
        else if (*i == "k"+axes->getNameForTran()) ktran.reset(py::extract<dcomplex>(kwargs[*i]));
        else throw TypeError("%2%() got unexpected keyword argument '%1%'", *i, name);
    }
    if (lambda) self->setWavelength(*lambda);
    if (neff) self->setKlong(*neff * self->getK0());
    if (ktran) self->setKtran(*ktran);
}


DataVectorWrap<const Tensor3<dcomplex>,2> FourierReflection2D_getRefractiveIndexProfile(FourierReflection2D& self,
                const shared_ptr<RectilinearMesh2D>& dst_mesh, InterpolationMethod interp=INTERPOLATION_DEFAULT) {
    return DataVectorWrap<const Tensor3<dcomplex>,2>(self.getRefractiveIndexProfile(*dst_mesh, interp), dst_mesh);
}

dcomplex FourierReflection2D_getDeterminant(py::tuple args, py::dict kwargs) {
    if (py::len(args) != 1)
        throw TypeError("determinant() takes exactly one non-keyword argument (%1% given)", py::len(args));
    FourierReflection2D* self = py::extract<FourierReflection2D*>(args[0]);
    FourierReflection2D_parseKeywords("determinant", self, kwargs);
    return self->getDeterminant();
}

PmlWrapper<FourierReflection2D> FourierReflection2D_getPML(FourierReflection2D* self) {
    return PmlWrapper<FourierReflection2D>(self, &self->pml);
}

double FourierReflection2D_Mode_Wavelength(const FourierReflection2D::Mode& mode) {
    return real(2e3 / mode.k0);
}
double FourierReflection2D_Mode_ModalLoss(const FourierReflection2D::Mode& mode) {
    return imag(2e4 * mode.k0);
}
double FourierReflection2D_Mode_Beta(const FourierReflection2D::Mode& mode) {
    return real(mode.beta);
}
double FourierReflection2D_Mode_Neff(const FourierReflection2D::Mode& mode) {
    return real(mode.beta / mode.k0);
}
double FourierReflection2D_Mode_KTran(const FourierReflection2D::Mode& mode) {
    return real(mode.ktran);
}
py::object FourierReflection2D_Mode_Symmetry(const FourierReflection2D::Mode& mode) {
    AxisNames* axes = getCurrentAxes();
    switch (mode.symmetry) {
        case ExpansionPW2D::SYMMETRIC_E_TRAN: return py::object("E"+axes->getNameForTran());
        case ExpansionPW2D::SYMMETRIC_E_LONG: return py::object("E"+axes->getNameForLong());
        default: return py::object();
    }
}



BOOST_PYTHON_MODULE(slab)
{
    {CLASS(FourierReflection2D, "FourierReflection2D",
        "Calculate optical modes and optical field distribution using Fourier slab method\n"
        " and reflection transfer in two-dimensional Cartesian space.")
        export_reflection_base(solver);
        PROVIDER(outNeff, "Effective index of the last computed mode");
        METHOD(find_mode, findMode, "Compute the mode near the specified effective index", "neff");
        RW_PROPERTY(wavelength, getWavelength, setWavelength, "Wavelength of the light");
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size");
        solver.add_property("symmetry", FourierReflection2D_getSymmetry, FourierReflection2D_setSymmetry, "Mode symmetry");
        RW_PROPERTY(polarization, getPolarization, setPolarization, "Mode polarization");
        RW_FIELD(refine, "Number of refinemnet points for refractive index averaging");
        solver.def("determinant", py::raw_function(FourierReflection2D_getDeterminant),
                   "Compute discontinuity matrix determinant");
        solver.def("get_refractive_index_profile", &FourierReflection2D_getRefractiveIndexProfile,
                   "Get profile of the expanded refractive index", (py::arg("mesh"), py::arg("interp")=INTERPOLATION_DEFAULT));
        solver.add_property("pml", py::make_function(&FourierReflection2D_getPML, py::with_custodian_and_ward_postcall<0,1>()), "Side PMLs");
        RO_PROPERTY(period, getPeriod, "Period for periodic structures");
        RO_FIELD(modes, "Computed modes");

        py::scope scope = solver;
        py_enum<ExpansionPW2D::Polarization>("Polarization", "Mode polarization.")
            .value("TE", ExpansionPW2D::TE)
            .value("TM", ExpansionPW2D::TM)
            .value("TEM", ExpansionPW2D::TEM)
        ;
        
        register_vector_of<FourierReflection2D::Mode>("Modes");

        py::class_<FourierReflection2D::Mode>("Mode", "Detailed information about the mode", py::no_init)
            .add_property("symmetry", &FourierReflection2D_Mode_Symmetry, "Mode horizontal symmetry")
            .def_readonly("polarization", &FourierReflection2D::Mode::polarization, "Mode polarization")
            .add_property("lam", &FourierReflection2D_Mode_Wavelength, "Mode wavelength [nm]")
            .add_property("wavelength", &FourierReflection2D_Mode_Wavelength, "Mode wavelength [nm]")
            .add_property("loss", &FourierReflection2D_Mode_ModalLoss, "Mode loss [1/cm]")
            .add_property("beta", &FourierReflection2D_Mode_Beta, "Mode longitudinal wavevector")
            .add_property("neff", &FourierReflection2D_Mode_Neff, "Mode longitudinal wavevector")
            .add_property("ktran", &FourierReflection2D_Mode_KTran, "Mode transverse wavevector")
            .def_readwrite("power", &FourierReflection2D::Mode::power, "Total power emitted into the mode")
        ;
    }

    {CLASS(FourierReflectionCyl, "FourierReflectionCyl",
        "Calculate optical modes and optical field distribution using Fourier slab method\n"
        " and reflection transfer in two-dimensional cylindrical geometry.")
        export_reflection_base(solver);
        METHOD(find_mode, findMode, "Compute the mode near the specified effective index", "neff"); //TODO
        RW_PROPERTY(size, getSize, setSize, "Orthogonal expansion size");
    }
}

