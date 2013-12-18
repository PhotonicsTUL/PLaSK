#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT.h"
using namespace plask::solvers::thermal;

namespace plask { namespace solvers { namespace thermal {

    std::string Convection__repr__(const Convection& self) {
        return "Convection(" + str(self.coeff) + "," + str(self.ambient) + ")";
    }

    std::string Radiation__repr__(const Radiation& self) {
        return "Radiation(" + str(self.emissivity) + "," + str(self.ambient) + ")";
    }

}}}



/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem)
{
    py_enum<Algorithm>("Algorithm", "Algorithms used for matrix factorization")
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py::class_<Convection>("Convection", "Convective boundary condition value", py::init<double,double>())
        .def_readwrite("coeff", &Convection::coeff)
        .def_readwrite("ambient", &Convection::ambient)
        .def("__repr__", &Convection__repr__)
    ;

    py::class_<Radiation>("Radiation", "Radiative boundary condition value", py::init<double,double>())
        .def_readwrite("emissivity", &Radiation::emissivity)
        .def_readwrite("ambient", &Radiation::ambient)
        .def("__repr__", &Radiation__repr__)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "Static2D", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "Receiver of heat density");
        solver.setattr("inHeatDensity", solver.attr("inHeat"));
        PROVIDER(outTemperature, "Provider of the temperatures");
        PROVIDER(outHeatFlux, "Provider of the heat flux");
        PROVIDER(outThermalConductivity, "Provider of the current thermal conductivity");
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

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "StaticCyl", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(err, getErr, "Maximum estimated error");
        RECEIVER(inHeat, "Receiver of heat density");
        solver.setattr("inHeatDensity", solver.attr("inHeat"));
        PROVIDER(outTemperature, "Provider of the temperatures");
        PROVIDER(outHeatFlux, "Provider of the heat flux");
        PROVIDER(outThermalConductivity, "Provider of the current thermal conductivity");
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

//     // Add methods to create classes using depreciate names
//     py::def("Fem2D", Fem2D, py::arg("name")="");
//     py::def("FemCyl", FemCyl, py::arg("name")="");
}
