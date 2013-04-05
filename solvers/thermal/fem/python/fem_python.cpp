#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT.h"
using namespace plask::solvers::thermal;

// static shared_ptr<FiniteElementMethodThermal2DSolver<Geometry2DCartesian>> Fem2D(const std::string& name) {
//     auto result = make_shared<FiniteElementMethodThermal2DSolver<Geometry2DCartesian>>(name);
//     result->writelog(LOG_WARNING, "'thermal.Fem2D' name is depreciated! Use 'thermal.Static2D' instead.");
//     return result;
// }
//
// static shared_ptr<FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>> FemCyl(const std::string& name) {
//     auto result = make_shared<FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>>(name);
//     result->writelog(LOG_WARNING, "'thermal.FemCyl' name is depreciated! Use 'thermal.StaticCyl' instead.");
//     return result;
// }

namespace plask { namespace solvers { namespace thermal {

    std::string Convection__repr__(const Convection& self) {
        return "Convection(" + str(self.mConvCoeff) + "," + str(self.mTAmb1) + ")";
    }

    std::string Radiation__repr__(const Radiation& self) {
        return "Radiation(" + str(self.mSurfEmiss) + "," + str(self.mTAmb2) + ")";
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
        .value("SLOW", ALGORITHM_SLOW)
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        //.value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<CorrectionType>("CorrectionType", "Types of the returned correction")
        .value("ABSOLUTE", CORRECTION_ABSOLUTE)
        .value("RELATIVE", CORRECTION_RELATIVE)
    ;

    py::class_<Convection>("Convection", "Convective boundary condition value", py::init<double,double>())
        .def_readwrite("coeff", &Convection::mConvCoeff)
        .def_readwrite("ambient", &Convection::mTAmb1)
        .def("__repr__", &Convection__repr__)
    ;

    py::class_<Radiation>("Radiation", "Radiative boundary condition value", py::init<double,double>())
        .def_readwrite("emissivity", &Radiation::mSurfEmiss)
        .def_readwrite("ambient", &Radiation::mTAmb2)
        .def("__repr__", &Radiation__repr__)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "Static2D", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsTCorr, "Maximum absolute correction for temperature");
        RO_PROPERTY(relcorr, getMaxRelTCorr, "Maximum relative correction for temperature");
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        solver.add_boundary_conditions("temperature_boundary", &__Class__::mTConst, "Boundary conditions for the constant temperature");
        solver.add_boundary_conditions("heatflux_boundary", &__Class__::mHFConst, "Boundary conditions for the constant heat flux");
        solver.add_boundary_conditions("convection_boundary", &__Class__::mConvection, "Convective boundary conditions");
        solver.add_boundary_conditions("radiation_boundary", &__Class__::mRadiation, "Radiative boundary conditions");
        RW_PROPERTY(inittemp, getTInit, setTInit, "Initial temperature"); // read-write property
        RW_PROPERTY(corrlim, getTCorrLim, setTCorrLim, "Limit for the temperature updates"); // read-write property
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
    }

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "StaticCyl", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsTCorr, "Maximum absolute correction for temperature");
        RO_PROPERTY(relcorr, getMaxRelTCorr, "Maximum relative correction for temperature");
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        solver.add_boundary_conditions("temperature_boundary", &__Class__::mTConst, "Boundary conditions for the constant temperature");
        solver.add_boundary_conditions("heatflux_boundary", &__Class__::mHFConst, "Boundary conditions for the constant heat flux");
        solver.add_boundary_conditions("convection_boundary", &__Class__::mConvection, "Convective boundary conditions");
        solver.add_boundary_conditions("radiation_boundary", &__Class__::mRadiation, "Radiative boundary conditions");
        RW_PROPERTY(inittemp, getTInit, setTInit, "Initial temperature"); // read-write property
        RW_PROPERTY(corrlim, getTCorrLim, setTCorrLim, "Limit for the temperature updates"); // read-write property
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
    }

//     // Add methods to create classes using depreciate names
//     py::def("Fem2D", Fem2D, py::arg("name")="");
//     py::def("FemCyl", FemCyl, py::arg("name")="");
}
