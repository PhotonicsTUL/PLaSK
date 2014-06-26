#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "iterative_matrix.h"
#include "gauss_matrix.h"

namespace plask { namespace solvers { namespace thermal {

/// Boundary condition: convection
struct Convection
{
    double coeff;   ///< convection coefficient [W/(m^2*K)]
    double ambient; ///< ambient temperature [K]
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
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API FiniteElementMethodThermal2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    int size;         ///< Number of columns in the main matrix

    int loopno;         ///< Number of completed loops
    double maxT;        ///< Maximum temperature recorded
    double toterr;      ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)

    DataVector<double> temperatures;           ///< Computed temperatures

    DataVector<Vec<2,double>> mHeatFluxes;      ///< Computed (only when needed) heat fluxes on our own mesh

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>,Radiation>& bradiation
                  );

    /// Update stored temperatures and calculate corrections
    double saveTemperatures(DataVector<double>& T);

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix solver
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

  public:

    // Boundary conditions
    BoundaryConditions<RectangularMesh<2>,double> temperature_boundary;      ///< Boundary condition of constant temperature [K]
    BoundaryConditions<RectangularMesh<2>,double> heatflux_boundary;         ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectangularMesh<2>,Convection> convection_boundary;   ///< Boundary condition of convection
    BoundaryConditions<RectangularMesh<2>,Radiation> radiation_boundary;     ///< Boundary condition of radiation

    typename ProviderFor<Temperature, Geometry2DType>::Delegate outTemperature;

    typename ProviderFor<HeatFlux, Geometry2DType>::Delegate outHeatFlux;

    typename ProviderFor<ThermalConductivity, Geometry2DType>::Delegate outThermalConductivity;

    ReceiverFor<Heat, Geometry2DType> inHeat;

    double maxerr;          ///< Maximum temperature correction accepted as convergence
    double inittemp;        ///< Initial temperature

    Algorithm algorithm;   ///< Factorization algorithm to use

    double itererr;        ///< Allowed residual iteration for iterative method
    size_t iterlim;        ///< Maximum nunber of iterations for iterative method
    size_t logfreq;        ///< Frequency of iteration progress reporting

    /**
     * Run temperature calculations
     * \return max correction of temperature against the last call
     **/
    double compute(int loops=1);

    /// Get max absolute correction for temperature
    double getErr() const { return toterr; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodThermal2DSolver(const std::string& name="");

    virtual std::string getClassName() const;

    ~FiniteElementMethodThermal2DSolver();

  protected:

    struct ThermalConductivityData: public LazyDataImpl<Tensor2<double>> {
        const FiniteElementMethodThermal2DSolver* solver;
        shared_ptr<RectangularMesh<2>> element_mesh;
        WrappedMesh<2> target_mesh;
        LazyData<double> temps;
        ThermalConductivityData(const FiniteElementMethodThermal2DSolver* solver, const shared_ptr<const MeshD<2>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const;
        std::size_t size() const;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) const;

    const LazyData<Vec<2>> getHeatFluxes(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method) const;

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage) {
        // boundary conditions of the first kind
        for (auto cond: bvoltage) {
            for (auto r: cond.place) {
                A(r,r) = 1.;
                double val = B[r] = cond.value;
                size_t start = (r > A.kd)? r-A.kd : 0;
                size_t end = (r + A.kd < A.size)? r+A.kd+1 : A.size;
                for(size_t c = start; c < r; ++c) {
                    B[c] -= A(r,c) * val;
                    A(r,c) = 0.;
                }
                for(size_t c = r+1; c < end; ++c) {
                    B[c] -= A(r,c) * val;
                    A(r,c) = 0.;
                }
            }
        }
    }

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage) {
        // boundary conditions of the first kind
        for (auto cond: bvoltage) {
            for (auto r: cond.place) {
                double* rdata = A.data + LDA*r;
                *rdata = 1.;
                double val = B[r] = cond.value;
                // below diagonal
                for (ptrdiff_t i = 4; i > 0; --i) {
                    ptrdiff_t c = r - A.bno[i];
                    if (c >= 0) {
                        B[c] -= A.data[LDA*c+i] * val;
                        A.data[LDA*c+i] = 0.;
                    }
                }
                // above diagonal
                for (ptrdiff_t i = 1; i < 5; ++i) {
                    ptrdiff_t c = r + A.bno[i];
                    if (c < A.size) {
                        B[c] -= rdata[i] * val;
                        rdata[i] = 0.;
                    }
                }
            }
        }
    }

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(int loops=1);

};

}} //namespaces

template <> inline solvers::thermal::Convection parseBoundaryValue<solvers::thermal::Convection>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Convection(tag_with_value.requireAttribute<double>("coeff"), tag_with_value.requireAttribute<double>("ambient"));
}

template <> inline solvers::thermal::Radiation parseBoundaryValue<solvers::thermal::Radiation>(const XMLReader& tag_with_value)
{
    return solvers::thermal::Radiation(tag_with_value.requireAttribute<double>("emissivity"), tag_with_value.requireAttribute<double>("ambient"));
}

} // namespace plask

#endif

