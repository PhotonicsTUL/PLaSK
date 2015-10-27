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

        "%1%(name=\"\")\n\n"

        "Finite element drift-diffusion electrical solver for 2D %2% geometry."

        , name, geoname).c_str(), py::init<std::string>(py::arg("name")=""));
    METHOD(compute, compute, "Run drift-diffusion calculations", py::arg("loops")=0);
    /*METHOD(get_total_current, getTotalCurrent, "Get total current flowing through active region [mA]", py::arg("nact")=0);
    RO_PROPERTY(err, getErr, "Maximum estimated error");*/
    //RO_PROPERTY(errPsi0, getErrPsi0, "Maximum estimated error for potential at U = 0 V"); czy to potrzebne?
    //RO_PROPERTY(errPsi, getErrPsi, "Maximum estimated error for potential"); czy to potrzebne?
    //RO_PROPERTY(errFn, getErrFn, "Maximum estimated error for quasi-Fermi energy level for electrons"); czy to potrzebne?
    //RO_PROPERTY(errFp, getErrFp, "Maximum estimated error for quasi-Fermi energy level for holes"); czy to potrzebne?
    /*RECEIVER(inWavelength, "It is required only if :attr:`heat` is equal to *wavelength*.");
    RECEIVER(inTemperature, "");*/
    PROVIDER(outPotential, "");
    PROVIDER(outQuasiFermiEnergyLevelForElectrons, "");
    PROVIDER(outQuasiFermiEnergyLevelForHoles, "");
    PROVIDER(outConductionBandEdge, "");
    PROVIDER(outValenceBandEdge, "");
    /*PROVIDER(outPotential, "");
    PROVIDER(outCurrentDensity, "");
    PROVIDER(outHeat, "");
    PROVIDER(outConductivity, "");*/
    BOUNDARY_CONDITIONS(voltage_boundary, "Boundary conditions of the first kind (constant potential)");
    RW_FIELD(maxerrPsiI, "Limit for the initial potential updates");
    RW_FIELD(maxerrPsi0, "Limit for the potential at U = 0 V updates");
    RW_FIELD(maxerrPsi, "Limit for the potential updates");
    RW_FIELD(maxerrFn, "Limit for the quasi-Fermi energy level for electrons updates");
    RW_FIELD(maxerrFp, "Limit for the quasi-Fermi energy level for holes updates");
    RW_FIELD(iterlimPsiI, "Maximum number of iterations for iterative method for initial potential");
    RW_FIELD(iterlimPsi0, "Maximum number of iterations for iterative method for potential at U = 0 V");
    RW_FIELD(iterlimPsi, "Maximum number of iterations for iterative method for potential");
    RW_FIELD(iterlimFn, "Maximum number of iterations for iterative method for quasi-Fermi energy level for electrons");
    RW_FIELD(iterlimFp, "Maximum number of iterations for iterative method for quasi-Fermi energy level for holes");
    //RW_FIELD(maxerr, "Limit for the potential updates");
    RW_FIELD(algorithm, "Chosen matrix factorization algorithm");
    /*solver.setattr("outVoltage", solver.attr("outPotential"));
    RW_FIELD(itererr, "Allowed residual iteration for iterative method");
    RW_FIELD(iterlim, "Maximum number of iterations for iterative method");
    RW_FIELD(logfreq, "Frequency of iteration progress reporting");
    METHOD(get_electrostatic_energy, getTotalEnergy,
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

    register_drift_diffusion_solver<Geometry2DCartesian>("DriftDiffusion2D", "Cartesian");

    register_drift_diffusion_solver<Geometry2DCylindrical>("DriftDiffusionCyl", "cylindrical");
}

