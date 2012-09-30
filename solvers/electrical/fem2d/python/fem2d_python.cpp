#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femV.h"
using namespace plask::solvers::electrical;

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(fem2d)
{
    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>, "CartesianFEM", "Finite Element electrical solver for 2D Cartesian Geometry.")
        METHOD(runCalc, "Run electrical calculations");
        RECEIVER(inTemperature, "Temperatures"); // receiver in the solver
        PROVIDER(outPotential, "Potentials"); // provider in the solver
        PROVIDER(outCurrentDensity, "CurrentDensities"); // provider in the solver
        PROVIDER(outHeatDensity, "HeatDensities"); // provider in the solver
        BOUNDARY_CONDITIONS(constV, mVconst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(getsetLoopLim, getLoopLim, setLoopLim, "Get and set max. number of loops"); // read-write property
        RW_PROPERTY(getsetVCorrLim, getVCorrLim, setVCorrLim, "Get and set limit for the potential updates"); // read-write property
        RW_PROPERTY(getsetVBigCorr, getVBigCorr, setVBigCorr, "Get and set initial value of the potential update"); // read-write property
        RW_PROPERTY(getsetBigNum, getBigNum, setBigNum, "Get and set big value for the first boundary condition"); // read-write property
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

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>, "CylindricalFEM", "Finite Element electrical solver for 2D Cylindrical Geometry.")
        METHOD(runCalc, "Run electrical calculations");
        RECEIVER(inTemperature, "Temperatures"); // receiver in the solver
        PROVIDER(outPotential, "Potentials"); // provider in the solver
        PROVIDER(outCurrentDensity, "CurrentDensities"); // provider in the solver
        PROVIDER(outHeatDensity, "HeatDensities"); // provider in the solver
        BOUNDARY_CONDITIONS(constV, mVconst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(getsetLoopLim, getLoopLim, setLoopLim, "Get and set max. number of loops"); // read-write property
        RW_PROPERTY(getsetVCorrLim, getVCorrLim, setVCorrLim, "Get and set limit for the potential updates"); // read-write property
        RW_PROPERTY(getsetVBigCorr, getVBigCorr, setVBigCorr, "Get and set initial value of the potential update"); // read-write property
        RW_PROPERTY(getsetBigNum, getBigNum, setBigNum, "Get and set big value for the first boundary condition"); // read-write property
    }
}

