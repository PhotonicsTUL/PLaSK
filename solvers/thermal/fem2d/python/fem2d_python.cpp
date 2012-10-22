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
BOOST_PYTHON_MODULE(fem2d)
{
    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCartesian>, "CartesianFEM", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(runCalc, "Run thermal calculations");
        METHOD(runSingleCalc, "Run single thermal calculations");
        METHOD(getMaxAbsTCorr, "Get max absolute correction for temperature");
        METHOD(getMaxRelTCorr, "Get max relative correction for temperature");
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        solver.add_boundary_conditions("Tconst", &__Class__::mTConst, "Boundary conditions for the constant temperature");
        solver.add_boundary_conditions("HFconst", &__Class__::mHFConst, "Boundary conditions for the constant heat flux");
        solver.add_boundary_conditions("convection", &__Class__::mConvection, "Convective boundary conditions");
        solver.add_boundary_conditions("radiation", &__Class__::mRadiation, "Radiative boundary conditions");
        RW_PROPERTY(loopLim, getLoopLim, setLoopLim, "Max. number of loops"); // read-write property
        RW_PROPERTY(TCorrLim, getTCorrLim, setTCorrLim, "Limit for the temperature updates"); // read-write property
        RW_PROPERTY(TBigCorr, getTBigCorr, setTBigCorr, "Initial value of the temperature update"); // read-write property
        RW_PROPERTY(bigNum, getBigNum, setBigNum, "Big value for the first boundary condition"); // read-write property
        RW_PROPERTY(TInit, getTInit, setTInit, "Initial temperature"); // read-write property
    }

    py::class_<Convection>("Convection", "Convective boundary condition value", py::init<double,double>())
        .def_readwrite("coeff", &Convection::mConvCoeff)
        .def_readwrite("ambient", &Convection::mTAmb1)
    ;

    py::class_<Radiation>("Radiation", "Radiative boundary condition value", py::init<double,double>())
        .def_readwrite("emissivity", &Radiation::mSurfEmiss)
        .def_readwrite("ambient", &Radiation::mTAmb2)
    ;

    {CLASS(FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>, "CylindricalFEM", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(runCalc, "Run thermal calculations");
        METHOD(runSingleCalc, "Run single thermal calculations");
        METHOD(getMaxAbsTCorr, "Get max absolute correction for temperature");
        METHOD(getMaxRelTCorr, "Get max relative correction for temperature");
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        solver.add_boundary_conditions("Tconst", &__Class__::mTConst, "Boundary conditions for the constant temperature");
        solver.add_boundary_conditions("HFconst", &__Class__::mHFConst, "Boundary conditions for the constant heat flux");
        solver.add_boundary_conditions("convection", &__Class__::mConvection, "Convective boundary conditions");
        solver.add_boundary_conditions("radiation", &__Class__::mRadiation, "Radiative boundary conditions");
        RW_PROPERTY(loopLim, getLoopLim, setLoopLim, "Max. number of loops"); // read-write property
        RW_PROPERTY(TCorrLim, getTCorrLim, setTCorrLim, "Limit for the temperature updates"); // read-write property
        RW_PROPERTY(TBigCorr, getTBigCorr, setTBigCorr, "Initial value of the temperature update"); // read-write property
        RW_PROPERTY(bigNum, getBigNum, setBigNum, "Big value for the first boundary condition"); // read-write property
        RW_PROPERTY(TInit, getTInit, setTInit, "Initial temperature"); // read-write property
    }
}

