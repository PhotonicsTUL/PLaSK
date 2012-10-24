#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femT.h"
using namespace plask::solvers::thermal;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem)
{
    py_enum<Algorithm>("Algorithm", "Algorithm used for matrix factorization")
        .value("SLOW", ALGORITHM_SLOW)
        .value("BLOCK", ALGORITHM_BLOCK)
        //.value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<CorrectionType>("CorrectionType", "Type of the returned correction")
        .value("ABSOLUTE", CORRECTION_ABSOLUTE)
        .value("RELATIVE", CORRECTION_RELATIVE)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "Fem2D", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(calculate, "Run thermal calculations", py::arg("loops")=1);
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
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Big value for the first boundary condition"); // read-write property
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
    }

    py::class_<Convection>("Convection", "Convective boundary condition value", py::init<double,double>())
        .def_readwrite("coeff", &Convection::mConvCoeff)
        .def_readwrite("ambient", &Convection::mTAmb1)
    ;

    py::class_<Radiation>("Radiation", "Radiative boundary condition value", py::init<double,double>())
        .def_readwrite("emissivity", &Radiation::mSurfEmiss)
        .def_readwrite("ambient", &Radiation::mTAmb2)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "FemCyl", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(calculate, "Run thermal calculations", py::arg("loops")=1);
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
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Big value for the first boundary condition"); // read-write property
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
    }
}

