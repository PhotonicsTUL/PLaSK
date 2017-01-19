#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../therm2d.h"
#include "../therm3d.h"
using namespace plask::thermal::tstatic;

namespace plask { namespace thermal { namespace tstatic {

    std::string Convection__repr__(const Convection& self) {
        return "Convection(" + str(self.coeff) + "," + str(self.ambient) + ")";
    }

    std::string Convection__str__(const Convection& self) {
        return str(self.coeff) + " (" + str(self.ambient) + "K)";
    }

    std::string Radiation__repr__(const Radiation& self) {
        return "Radiation(" + str(self.emissivity) + "," + str(self.ambient) + ")";
    }

    std::string Radiation__str__(const Radiation& self) {
        return str(self.emissivity) + " (" + str(self.ambient) + "K)";
    }

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

    py::class_<Convection>("Convection", "Convective boundary condition value.", py::init<double,double>())
        .def_readwrite("coeff", &Convection::coeff)
        .def_readwrite("ambient", &Convection::ambient)
        .def("__repr__", &Convection__repr__)
        .def("__str__", &Convection__str__)
    ;

    py::class_<Radiation>("Radiation", "Radiative boundary condition value.", py::init<double,double>())
        .def_readwrite("emissivity", &Radiation::emissivity)
        .def_readwrite("ambient", &Radiation::ambient)
        .def("__repr__", &Radiation__repr__)
        .def("__str__", &Radiation__str__)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "Static2D",
        "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, "Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, "Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, "Radiative boundary conditions");
        RW_FIELD(inittemp, "Initial temperature");
        RW_FIELD(maxerr, "Limit for the temperature updates");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("itererr", &__Class__::itererr, "Allowed residual iteration for iterative method");
        solver.def_readwrite("iterlim", &__Class__::iterlim ,"Maximum number of iterations for iterative method");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
    }

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "StaticCyl",
        "Finite element thermal solver for 2D cylindrical Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "");
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, "Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, "Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, "Radiative boundary conditions");
        RW_FIELD(inittemp, "Initial temperature");
        RW_FIELD(maxerr, "Limit for the temperature updates");
        solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("itererr", &__Class__::itererr, "Allowed residual iteration for iterative method");
        solver.def_readwrite("iterlim", &__Class__::iterlim ,"Maximum number of iterations for iterative method");
        solver.def_readwrite("logfreq", &__Class__::logfreq ,"Frequency of iteration progress reporting");
    }

    {CLASS(FiniteElementMethodThermal3DSolver, "Static3D", "Finite element thermal solver for 3D Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "");
        solver.setattr("inHeatDensity", solver.attr("inHeat"));
        PROVIDER(outTemperature, "");
        PROVIDER(outHeatFlux, "");
        PROVIDER(outThermalConductivity, "");
        BOUNDARY_CONDITIONS(temperature_boundary, "Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(heatflux_boundary, "Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection_boundary, "Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation_boundary, "Radiative boundary conditions");
        RW_FIELD(inittemp, "Initial temperature");
        RW_FIELD(maxerr, "Limit for the temperature updates");
        RW_PROPERTY(algorithm, getAlgorithm, setAlgorithm, "Chosen matrix factorization algorithm");
        RW_FIELD(itererr, "Allowed residual iteration for iterative method");
        RW_FIELD(iterlim, "Maximum number of iterations for iterative method");
        RW_FIELD(logfreq, "Frequency of iteration progress reporting");
    }
}
