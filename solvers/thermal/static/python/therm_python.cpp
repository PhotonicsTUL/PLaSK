#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../therm2d.h"
#include "../therm3d.h"
using namespace plask::thermal::tstatic;

namespace plask { namespace thermal { namespace tstatic {

template <typename T>
struct Bc {

    static const char* NAME;
    static const char* FIRST;

    inline static double& first(T& self);

    static void* convertible(PyObject* obj) {
        if (!PyDict_Check(obj)) return nullptr;
        return obj;
    }
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<Tensor2<double>>*)data)->storage.bytes;
        double first = py::extract<double>(PyDict_GetItemString(obj, FIRST));
        PyObject* pyambient = PyDict_GetItemString(obj, "ambient");
        double ambient = pyambient? py::extract<double>(pyambient) : 300.;
        new(storage) T(first, ambient);
        data->convertible = storage;
    }

    Bc(const char* doc) {
        py::converter::registry::push_back(&convertible, &construct, boost::python::type_id<T>());
        py::class_<T>(NAME, doc, py::init<double,double>())
            .def("__repr__", &Bc<T>::__repr__)
            .def("__str__", &Bc<T>::__str__)
            .def("__getitem__", &Bc<T>::__getitem__)
            .def("__setitem__", &Bc<T>::__setitem__)
        ;
    }

    static std::string __str__(T& self) {
        return str(first(self)) + " (" + str(self.ambient) + "K)";
    }

    static std::string __repr__(T& self) {
        return "{'" + std::string(FIRST) + "': " + str(first(self)) + ", 'ambient': " + str(self.ambient) + "}";
    }

    static double __getitem__(T& self, const std::string& key) {
        if (key == FIRST) return first(self);
        else if (key == "ambient") return self.ambient;
        else throw KeyError(key);
    }

    static void __setitem__(T& self, const std::string& key, double value) {
        if (key == FIRST) first(self) = value;
        else if (key == "ambient") self.ambient = value;
        else throw KeyError(key);
    }

};

template<> const char* Bc<Convection>::NAME = "Convection";
template<> const char* Bc<Convection>::FIRST = "coeff";
template<> double& Bc<Convection>::first(Convection& self) { return self.coeff; }

template<> const char* Bc<Radiation>::NAME = "Radiation";
template<> const char* Bc<Radiation>::FIRST = "emissivity";
template<> double& Bc<Radiation>::first(Radiation& self) { return self.emissivity; }

}}}


/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(static)
{
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    Bc<Convection>(u8"Convective boundary condition value.");
    Bc<Radiation>(u8"Radiative boundary condition value.");

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "Static2D",
        u8"Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, u8"Maximum estimated error");
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, u8"Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, u8"Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, u8"Radiative boundary conditions");
        RW_FIELD(inittemp, u8"Initial temperature");
        RW_FIELD(maxerr, u8"Limit for the temperature updates");
        RW_PROPERTY(include_empty, usingFullMesh, useFullMesh, "Should empty regions (e.g. air) be included into computation domain?");
        solver.def_readwrite("algorithm", &__Class__::algorithm, u8"Chosen matrix factorization algorithm");
        solver.def_readwrite("itererr", &__Class__::itererr, u8"Allowed residual iteration for iterative method");
        solver.def_readwrite("iterlim", &__Class__::iterlim, u8"Maximum number of iterations for iterative method");
        solver.def_readwrite("logfreq", &__Class__::logfreq, u8"Frequency of iteration progress reporting");
    }

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "StaticCyl",
        u8"Finite element thermal solver for 2D cylindrical Geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, u8"Maximum estimated error");
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, u8"Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, u8"Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, u8"Radiative boundary conditions");
        RW_FIELD(inittemp, u8"Initial temperature");
        RW_FIELD(maxerr, u8"Limit for the temperature updates");
        RW_PROPERTY(include_empty, usingFullMesh, useFullMesh, "Should empty regions (e.g. air) be included into computation domain?");
        RW_FIELD(algorithm, u8"Chosen matrix factorization algorithm");
        RW_FIELD(itererr, u8"Allowed residual iteration for iterative method");
        RW_FIELD(iterlim, u8"Maximum number of iterations for iterative method");
        RW_FIELD(logfreq, u8"Frequency of iteration progress reporting");
    }

    {CLASS(FiniteElementMethodThermal3DSolver, "Static3D", u8"Finite element thermal solver for 3D Geometry.")
        METHOD(compute, compute, u8"Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, u8"Maximum estimated error");
        RECEIVER(inHeat, "");
        solver.setattr("inHeatDensity", solver.attr("inHeat"));
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, u8"Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, u8"Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, u8"Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, u8"Radiative boundary conditions");
        RW_FIELD(inittemp, u8"Initial temperature");
        RW_FIELD(maxerr, u8"Limit for the temperature updates");
        RW_PROPERTY(include_empty, usingFullMesh, useFullMesh, "Should empty regions (e.g. air) be included into computation domain?");
        RW_PROPERTY(algorithm, getAlgorithm, setAlgorithm, u8"Chosen matrix factorization algorithm");
        RW_FIELD(itererr, u8"Allowed residual iteration for iterative method");
        RW_FIELD(iterlim, u8"Maximum number of iterations for iterative method");
        RW_FIELD(logfreq, u8"Frequency of iteration progress reporting");
    }
}
