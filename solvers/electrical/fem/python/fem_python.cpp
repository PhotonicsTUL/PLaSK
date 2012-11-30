#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../femV.h"
using namespace plask::solvers::electrical;

static shared_ptr<FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>> Fem2D(const std::string& name) {
    auto result = make_shared<FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>>(name);
    result->writelog(LOG_WARNING, "'electrical.Fem2D' name is depreciated! Use 'electrical.Beta2D' instead.");
    return result;
}

static shared_ptr<FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>> FemCyl(const std::string& name) {
    auto result = make_shared<FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>>(name);
    result->writelog(LOG_WARNING, "'electrical.FemCyl' name is depreciated! Use 'electrical.BetaCyl' instead.");
    return result;
}

static shared_ptr<SolverOver<Geometry2DCartesian>> DriftDiffusion2D(const std::string& name) {
    throw NotImplemented("DriftDiffusion2D: I want it to be implemented too!");
}

static shared_ptr<SolverOver<Geometry2DCylindrical>> DriftDiffusionCyl(const std::string& name) {
    throw NotImplemented("DriftDiffusionCyl: I want it to be implemented too!");
}

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
        .value("BLOCK", ALGORITHM_BLOCK)
        //.value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<CorrectionType>("CorrectionType", "Types of the returned correction")
        .value("ABSOLUTE", CORRECTION_ABSOLUTE)
        .value("RELATIVE", CORRECTION_RELATIVE)
    ;

    py_enum<HeatMethod>("HeatType", "Methods used for computing heats")
        .value("JOULES", HEAT_JOULES)
        .value("WAVELENGTH", HEAT_BANDGAP)
    ;

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>, "Beta2D", "Finite element thermal solver for 2D Cartesian Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Maximum absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Maximum relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specifying the bad-gap");
        RECEIVER(inTemperature, "Temperatures");
        PROVIDER(outPotential, "Potentials");
        PROVIDER(outCurrentDensity, "CurrentDensities");
        PROVIDER(outHeatDensity, "HeatDensities");
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Limit for the potential updates");
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Big value for the boundary condition");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::mHeatMethod, "Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient");
        RW_PROPERTY(js, getJs, setJs, "Reverse current [A/m²]");
        RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
        RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
        RW_PROPERTY(pnjcond, getCondJunc0, setCondJunc0, "Conductivity of the n-contact");
    }

    {CLASS(FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>, "BetaCyl", "Finite element thermal solver for 2D Cylindrical Geometry.")
        METHOD(compute, compute, "Run thermal calculations", py::arg("loops")=0);
        RO_PROPERTY(abscorr, getMaxAbsVCorr, "Maximum absolute correction for potential");
        RO_PROPERTY(relcorr, getMaxRelVCorr, "Maximum relative correction for potential");
        RECEIVER(inWavelength, "Wavelength specifying the bad-gap");
        RECEIVER(inTemperature, "Temperatures");
        PROVIDER(outPotential, "Potentials");
        PROVIDER(outCurrentDensity, "CurrentDensities");
        PROVIDER(outHeatDensity, "HeatDensities");
        solver.add_boundary_conditions("voltage_boundary", &__Class__::mVConst, "Boundary conditions of the first kind (constant potential)");
        RW_PROPERTY(corrlim, getVCorrLim, setVCorrLim, "Limit for the potential updates");
        solver.def_readwrite("corrtype", &__Class__::mCorrType, "Type of returned correction");
        RW_PROPERTY(bignum, getBigNum, setBigNum, "Big value for the boundary condition");
        solver.def_readwrite("algorithm", &__Class__::mAlgorithm, "Chosen matrix factorization algorithm");
        solver.def_readwrite("heat", &__Class__::mHeatMethod, "Chosen method used for computing heats");
        RW_PROPERTY(beta, getBeta, setBeta, "Junction coefficient");
        RW_PROPERTY(js, getJs, setJs, "Reverse current [A/m²]");
        RW_PROPERTY(pcond, getCondPcontact, setCondPcontact, "Conductivity of the p-contact");
        RW_PROPERTY(ncond, getCondNcontact, setCondNcontact, "Conductivity of the n-contact");
        RW_PROPERTY(pnjcond, getCondJunc0, setCondJunc0, "Initial conductivity of the p-n junction");
    }

    py::def("DriftDiffusion2D", DriftDiffusion2D, py::arg("name")="");
    py::def("DriftDiffusionCyl", DriftDiffusionCyl, py::arg("name")="");

    // Add methods to create classes using depreciate names
    py::def("Fem2D", Fem2D, py::arg("name")="");
    py::def("FemCyl", FemCyl, py::arg("name")="");
}

