#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "gauss_matrix.h"
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
    ALGORITHM_CHOLESKY, ///< block algorithm (thrice faster, however a little prone to failures)
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< iterative algorithm using preconditioned conjugate gradient method
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct PLASK_SOLVER_API FiniteElementMethodThermal3DSolver: public SolverWithMesh<Geometry3D, RectangularMesh<3>> {

  protected:

    Algorithm algorithm;   ///< Factorization algorithm to use

    int loopno;         ///< Number of completed loops
    double maxT;        ///< Maximum temperature recorded
    double toterr;      ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)

    DataVector<double> temperatures;            ///< Computed temperatures

    DataVector<double> thickness;               ///< Thicknesses of the layers

    DataVector<Vec<3,double>> fluxes;           ///< Computed (only when needed) heat fluxes on our own mesh

    /**
     * Set stiffness matrix and load vector
     * \param[out] A matrix to fill-in
     * \param[out] B load vector
     * \param btemperature boundary conditions: constant temperature
     * \param bheatflux boundary conditions: constant heat flux
     * \param bconvection boundary conditions: convention
     * \param bradiation boundary conditions: radiation
     **/
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>,Radiation>& bradiation
                  );

    /**
     * Apply boundary conditions of the first kind
     */
    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<3>,double>& btemperature);

    /// Update stored temperatures and calculate corrections
    double saveTemperatures(DataVector<double>& T);

    /// Create 3D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix solver for the block cholesky algorithm
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver for the block gauss algorithm
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver for the iterative algorithm
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(int loops=1);

  public:

    double inittemp;    ///< Initial temperature
    double maxerr;     ///< Maximum temperature correction accepted as convergence

    double itererr;     ///< Allowed residual iteration for iterative method
    size_t iterlim;     ///< Maximum nunber of iterations for iterative method

    size_t logfreq;   ///< Frequency of iteration progress reporting

    // Boundary conditions
    BoundaryConditions<RectangularMesh<3>,double> temperature_boundary;      ///< Boundary condition of constant temperature [K]
    BoundaryConditions<RectangularMesh<3>,double> heatflux_boundary;         ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectangularMesh<3>,Convection> convection_boundary;   ///< Boundary condition of convection
    BoundaryConditions<RectangularMesh<3>,Radiation> radiation_boundary;     ///< Boundary condition of radiation

    typename ProviderFor<Temperature,Geometry3D>::Delegate outTemperature;

    typename ProviderFor<HeatFlux,Geometry3D>::Delegate outHeatFlux;

    typename ProviderFor<ThermalConductivity,Geometry3D>::Delegate outThermalConductivity;

    ReceiverFor<Heat,Geometry3D> inHeat;

    /**
     * Run temperature calculations
     * \param loops maximum number of loops to run
     * \return max correction of temperature against the last call
     **/
    double compute(int loops=1);

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermal3DSolver(const std::string& name="");

    virtual std::string getClassName() const { return "thermal.Static3D"; }

    ~FiniteElementMethodThermal3DSolver();

    /// Get max absolute correction for temperature
    double getErr() const { return toterr; }

    /// \return current algorithm
    Algorithm getAlgorithm() const { return algorithm; }

    /**
     * Set algorithm
     * \param alg new algorithm
     */
    void setAlgorithm(Algorithm alg);

  protected:

    struct ThermalConductivityData: public LazyDataImpl<Tensor2<double>> {
        const FiniteElementMethodThermal3DSolver* solver;
        WrappedMesh<3> target_mesh;
        LazyData<double> temps;
        ThermalConductivityData(const FiniteElementMethodThermal3DSolver* solver, const shared_ptr<const MeshD<3>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const;
        std::size_t size() const;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method) const;

    const LazyData<Vec<3>> getHeatFluxes(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<3>>& dst_mesh, InterpolationMethod method);
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

