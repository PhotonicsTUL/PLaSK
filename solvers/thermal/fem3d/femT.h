#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "iterative_matrix.h"

namespace plask { namespace solvers { namespace thermal3d {

/// Boundary condition: convection
struct Convection
{
    double coeff;       ///< convection coefficient [W/(m^2*K)]
    double ambient;     ///< ambient temperature [K]
    Convection(double coeff, double amb): coeff(coeff), ambient(amb) {}
    Convection() = default;
};

/// Boundary condition: radiation
struct Radiation
{
    double emissivity;  ///< surface emissivity [-]
    double ambient;     ///< ambient temperature [K]
    Radiation(double emiss, double amb): emissivity(emiss), ambient(amb) {}
    Radiation() = default;
};


/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_BLOCK,    ///< block algorithm (thrice faster, however a little prone to failures)
    ALGORITHM_ITERATIVE ///< iterative algorithm using preconditioned conjugate gradient method
};

/// Type of the returned correction
enum CorrectionType {
    CORRECTION_ABSOLUTE,    ///< absolute correction is used
    CORRECTION_RELATIVE     ///< relative correction is used
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct FiniteElementMethodThermal3DSolver: public SolverWithMesh<Geometry3D, RectilinearMesh3D> {

  protected:

    Algorithm algorithm;   ///< Factorization algorithm to use

    int loopno;         ///< Number of completed loops
    double maxT;        ///< Maximum temperature recorded
    double corr;        ///< Maximum absolute temperature correction (useful for calculations with internal loops)

    DataVector<double> temperatures;            ///< Computed temperatures

    DataVector<Vec<3,double>> fluxes;           ///< Computed (only when needed) heat fluxes on our own mesh

    /**
     * Set stiffness matrix and load vector
     * \param[out] A matrix to fill-in
     * \param[out] B load vector
     * \param constT boundary conditions: constant temperature
     * \param constHF boundary conditions: constant heat flux
     * \param convection boundary conditions: convention
     * \param radiation boundary conditions: radiation
     **/
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh3D,Radiation>& bradiation
                  );

    /**
     * Apply boundary conditions of the first kind
     */
    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& btemperature);

    /// Update stored temperatures and calculate corrections
    void saveTemperatures(DataVector<double>& T);

    /// Create 3D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix solver for block algorithm
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver for iterative algorithm
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    template <typename MatrixT>
    double doCompute(int loops=1);

  public:

    double inittemp;    ///< Initial temperature
    double corrlim;     ///< Maximum temperature correction accepted as convergence
    double abscorr;     ///< Maximum absolute temperature correction (useful for single calculations managed by external python script)
    double relcorr;     ///< Maximum relative temperature correction (useful for single calculations managed by external python script)
    CorrectionType corrtype; ///< Type of the returned correction

    double itererr;     ///< Allowed residual iteration for iterative method
    unsigned itermax;   ///< Maximum nunber of iterations for iterative method

    bool equilibrate;      ///< Should the matrix system be equilibrated before solving?

    // Boundary conditions
    BoundaryConditions<RectilinearMesh3D,double> temperature_boundary;      ///< Boundary condition of constant temperature [K]
    BoundaryConditions<RectilinearMesh3D,double> heatflux_boundary;         ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectilinearMesh3D,Convection> convection_boundary;   ///< Boundary condition of convection
    BoundaryConditions<RectilinearMesh3D,Radiation> radiation_boundary;     ///< Boundary condition of radiation

    typename ProviderFor<Temperature,Geometry3D>::Delegate outTemperature;

    typename ProviderFor<HeatFlux3D,Geometry3D>::Delegate outHeatFlux;

    ReceiverFor<HeatDensity,Geometry3D> inHeatDensity;

    /**
     * Run temperature calculations
     * \param looplim maximum number of loops to run
     * \return max correction of temperature against the last call
     **/
    double compute(int loops=1);

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermal3DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "thermal.Static3D"; }

    ~FiniteElementMethodThermal3DSolver();

    /// \return current algorithm
    Algorithm getAlgorithm() const { return algorithm; }

    /**
     * Set algorithm
     * \param alg new algorithm
     */
    void setAlgorithm(Algorithm alg);

  protected:

    DataVector<const double> getTemperatures(const MeshD<3>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<3>> getHeatFluxes(const MeshD<3>& dst_mesh, InterpolationMethod method);
};

}} //namespaces

template <> inline solvers::thermal3d::Convection parseBoundaryValue<solvers::thermal3d::Convection>(const XMLReader& tag_with_value)
{
    return solvers::thermal3d::Convection(tag_with_value.requireAttribute<double>("coeff"), tag_with_value.requireAttribute<double>("ambient"));
}

template <> inline solvers::thermal3d::Radiation parseBoundaryValue<solvers::thermal3d::Radiation>(const XMLReader& tag_with_value)
{
    return solvers::thermal3d::Radiation(tag_with_value.requireAttribute<double>("emissivity"), tag_with_value.requireAttribute<double>("ambient"));
}

} // namespace plask

#endif

