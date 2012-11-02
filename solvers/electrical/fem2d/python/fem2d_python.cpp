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
    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>, "Fem2D", "Finite Element electrical solver for 2D Cartesian Geometry.")
        METHOD(compute, runCalc, "Run electrical calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Max absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Max relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specyfiing the bad-gap");
        RECEIVER(inTemperature, "Temperatures"); // receiver in the solver
        PROVIDER(outPotential, "Potentials"); // provider in the solver
        PROVIDER(outCurrentDensity, "CurrentDensities"); // provider in the solver
        PROVIDER(outHeatDensity, "HeatDensities"); // provider in the solver
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Get and set limit for the potential updates"); // read-write property
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Get and set big value for the first boundary condition"); // read-write property
    }

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>, "FemCyl", "Finite Element electrical solver for 2D Cylindrical Geometry.")
        METHOD(compute, runCalc, "Run electrical calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Max absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Max relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specyfiing the bad-gap");
        RECEIVER(inTemperature, "Temperatures"); // receiver in the solver
        PROVIDER(outPotential, "Potentials"); // provider in the solver
        PROVIDER(outCurrentDensity, "CurrentDensities"); // provider in the solver
        PROVIDER(outHeatDensity, "HeatDensities"); // provider in the solver
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Get and set limit for the potential updates"); // read-write property
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Get and set big value for the first boundary condition"); // read-write property
    }
}

