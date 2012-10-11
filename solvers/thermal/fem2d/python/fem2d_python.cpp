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
        METHOD(getMaxAbsTCorr, "Get max absolute correction for temperature");
        METHOD(getMaxRelTCorr, "Get max relative correction for temperature");
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        BOUNDARY_CONDITIONS(Tconst, mTConst, "Boundary conditions for the constant temperature");
        BOUNDARY_CONDITIONS(HFconst, mHFConst, "Boundary conditions for the constant heat flux");
        BOUNDARY_CONDITIONS(convection, mConvection, "Convective boundary conditions");
        BOUNDARY_CONDITIONS(radiation, mRadiation, "Radiative boundary conditions");
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
        RECEIVER(inHeatDensity, "HeatDensities"); // receiver in the solver
        PROVIDER(outTemperature, "Temperatures"); // provider in the solver
        PROVIDER(outHeatFlux, "HeatFluxes"); // provider in the solver
        BOUNDARY_CONDITIONS(Tconst, mTConst, "Boundary conditions of the first kind (constant temperature)");
        RW_PROPERTY(loopLim, getLoopLim, setLoopLim, "Max. number of loops"); // read-write property
        RW_PROPERTY(TCorrLim, getTCorrLim, setTCorrLim, "Limit for the temperature updates"); // read-write property
        RW_PROPERTY(TBigCorr, getTBigCorr, setTBigCorr, "Initial value of the temperature update"); // read-write property
        RW_PROPERTY(bigNum, getBigNum, setBigNum, "Big value for the first boundary condition"); // read-write property
        RW_PROPERTY(TInit, getTInit, setTInit, "Initial temperature"); // read-write property
    }
}

