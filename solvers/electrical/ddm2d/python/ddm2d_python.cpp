#include <cmath>
#include <plask/python.hpp>
using namespace plask;
using namespace plask::python;

#include "../ddm2d.h"
using namespace plask::solvers::drift_diffusion;

template <typename GeometryT>
inline static void register_drift_diffusion_solver(const char* name, const char* geoname)
{
    typedef DriftDiffusionModel2DSolver<GeometryT>  __Class__;
    ExportSolver<DriftDiffusionModel2DSolver<GeometryT>> solver(name, format(

        "{0}(name=\"\")\n\n"

        "Finite element drift-diffusion electrical solver for 2D {1} geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    METHOD(compute, compute, "Run drift-diffusion calculations", py::arg("loops")=0);
    METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
    //METHOD(integrate_current, integrateCurrent, "Integrate vertical total current at certain level [mA]", py::arg("vindex"), py::arg("onlyactive")=false);
    /*RO_PROPERTY(err, getErr, "Maximum estimated error");*/
    RECEIVER(inTemperature, "");
    PROVIDER(outPotential, "");
    PROVIDER(outQuasiFermiEnergyLevelForElectrons, "");
    PROVIDER(outQuasiFermiEnergyLevelForHoles, "");
    PROVIDER(outConductionBandEdge, "");
    PROVIDER(outValenceBandEdge, "");
    PROVIDER(outCurrentDensityForElectrons, "");
    PROVIDER(outCurrentDensityForHoles, "");
    PROVIDER(outElectronConcentration, "");
    PROVIDER(outHoleConcentration, "");
    PROVIDER(outHeat, "");
    /*PROVIDER(outConductivity, "");*/
    BOUNDARY_CONDITIONS(voltage_boundary, "Boundary conditions of the first kind (constant potential)");
    solver.def_readwrite("maxerrVi", &__Class__::maxerrPsiI, "Limit for the initial potential estimate updates");
    solver.def_readwrite("maxerrV0", &__Class__::maxerrPsi0, "Limit for the built-in potential updates");
    solver.def_readwrite("maxerrV", &__Class__::maxerrPsi, "Limit for the potential updates");
    solver.def_readwrite("maxerrFn", &__Class__::maxerrFn, "Limit for the electrons quasi-Fermi level updates");
    solver.def_readwrite("maxerrFp", &__Class__::maxerrFp, "Limit for the holes quasi-Fermi level updates");
    solver.def_readwrite("loopsVi", &__Class__::loopsPsiI, "Loops limit for the initial potential estimate");
    solver.def_readwrite("loopsV0", &__Class__::loopsPsi0, "Loops limit for the built-in potential");
    solver.def_readwrite("loopsV", &__Class__::loopsPsi, "Loops limit for the potential");
    solver.def_readwrite("loopsFn", &__Class__::loopsFn, "Loops limit for the electrons quasi-Fermi level");
    solver.def_readwrite("loopsFp", &__Class__::loopsFp, "Loops limit for the holes quasi-Fermi level");
    solver.def_readwrite("algorithm", &__Class__::algorithm, "Chosen matrix factorization algorithm");
    solver.def_readwrite("itererr", &__Class__::itererr, "Allowed residual iteration for iterative method");
    solver.def_readwrite("iterlim", &__Class__::iterlim, "Maximum number of iterations for iterative method");
    solver.def_readwrite("logfreq", &__Class__::logfreq, "Frequency of iteration progress reporting");
    solver.def_readwrite("Rsrh", &__Class__::mRsrh, "True if SRH recombination is taken into account");
    solver.def_readwrite("Rrad", &__Class__::mRrad, "True if radiative recombination is taken into account");
    solver.def_readwrite("Raug", &__Class__::mRaug, "True if Auger recombination is taken into account");
    solver.def_readwrite("Pol", &__Class__::mPol, "True if polarization effects are taken into account");
    solver.def_readwrite("FullIon", &__Class__::mFullIon, "True if dopants are completely ionized");
    solver.def_readwrite("SchottkyP", &__Class__::mSchottkyP, "Schottky barrier for p-type constact");
    solver.def_readwrite("SchottkyN", &__Class__::mSchottkyN, "Schottky barrier for n-type constact");
    /*METHOD(get_electrostatic_energy, getTotalEnergy,
           "Get the energy stored in the electrostatic field in the analyzed structure.\n\n"
           "Return:\n"
           "    Total electrostatic energy [J].\n"
    );
    METHOD(get_capacitance, getCapacitance,
           "Get the structure capacitance.\n\n"
           "Return:\n"
           "    Total capacitance [pF].\n\n"
           "Note:\n"
           "    This method can only be used it there are exactly two boundary conditions\n"
           "    specifying the voltage. Otherwise use :meth:`get_electrostatic_energy` to\n"
           "    obtain the stored energy :math:`W` and compute the capacitance as:\n"
           "    :math:`C = 2 \\, W / U^2`, where :math:`U` is the applied voltage.\n"
    );
    METHOD(get_total_heat, getTotalHeat,
           "Get the total heat produced by the current flowing in the structure.\n\n"
           "Return:\n"
           "    Total produced heat [mW].\n"
    );*/
}

/**
 * Initialization of your solver class to Python
 *
 * The \a solver_name should be changed to match the name of the directory with our solver
 * (the one where you have put CMakeLists.txt). It will be visible from user interface under this name.
 */
BOOST_PYTHON_MODULE(ddm2d)
{
    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE)
    ;

    py_enum<Stat>()
        .value("MAXWELL_BOLTZMANN", STAT_MB)
        .value("FERMI_DIRAC", STAT_FD)
    ;

    py_enum<ContType>()
        .value("OHMIC", OHMIC)
        .value("SCHOTTKY", SCHOTTKY)
    ;

    register_drift_diffusion_solver<Geometry2DCartesian>("DriftDiffusion2D", "Cartesian");

    register_drift_diffusion_solver<Geometry2DCylindrical>("DriftDiffusionCyl", "cylindrical");
}

