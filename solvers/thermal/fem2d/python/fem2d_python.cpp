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
/*
        METHOD(method_name, "Short documentation", "name_or_argument_1", arg("name_of_argument_2")=default_value_of_arg_2, ...);
        RO_FIELD(field_name, "Short documentation"); // read-only field
        RW_FIELD(field_name, "Short documentation"); // read-write field
        RO_PROPERTY(python_property_name, get_method_name, "Short documentation"); // read-only property
        RW_PROPERTY(python_property_name, get_method_name, set_method_name, "Short documentation"); // read-write property
        RECEIVER(inReceiver, "Short documentation"); // receiver in the solver
        PROVIDER(outProvider, "Short documentation"); // provider in the solver
*/
    }

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

