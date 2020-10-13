#ifndef PLASK__MODULE_THERMAL_THERM2D_H
#define PLASK__MODULE_THERMAL_THERM2D_H

#include <plask/plask.hpp>

#include "common.hpp"
#include "block_matrix.hpp"
#include "iterative_matrix2d.hpp"
#include "gauss_matrix.hpp"

namespace plask { namespace thermal { namespace tstatic {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API ThermalFem2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    /// Masked mesh
    plask::shared_ptr<RectangularMaskedMesh2D> maskedMesh = plask::make_shared<RectangularMaskedMesh2D>();

    int loopno;         ///< Number of completed loops
    double maxT;        ///< Maximum temperature recorded
    double toterr;      ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)

    DataVector<double> temperatures;            ///< Computed temperatures

    DataVector<double> thickness;               ///< Thicknesses of the layers

    DataVector<Vec<2,double>> fluxes;           ///< Computed (only when needed) heat fluxes on our own mesh

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bheatflux,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Convection>& bconvection,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,Radiation>& bradiation
                  );

    /// Setup matrix
    template <typename MatrixT>
    MatrixT makeMatrix();

    /// Update stored temperatures and calculate corrections
    double saveTemperatures(DataVector<double>& T);

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix solver
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(SparseBandMatrix2D& A, DataVector<double>& B);

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

  public:

    // Boundary conditions
    BoundaryConditions<RectangularMesh<2>::Boundary,double> temperature_boundary;      ///< Boundary condition of constant temperature [K]
    BoundaryConditions<RectangularMesh<2>::Boundary,double> heatflux_boundary;         ///< Boundary condition of constant heat flux [W/m^2]
    BoundaryConditions<RectangularMesh<2>::Boundary,Convection> convection_boundary;   ///< Boundary condition of convection
    BoundaryConditions<RectangularMesh<2>::Boundary,Radiation> radiation_boundary;     ///< Boundary condition of radiation

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

    /// Are we using full mesh?
    bool usingFullMesh() const { return use_full_mesh; }
    /// Set whether we should use full mesh
    void useFullMesh(bool val) {
        use_full_mesh = val;
        this->invalidate();
    }

    /**
     * Run temperature calculations
     * \return max correction of temperature against the last call
     **/
    double compute(int loops=1);

    /// Get max absolute correction for temperature
    double getErr() const { return toterr; }

    void loadConfiguration(XMLReader& source, Manager& manager) override; // for solver configuration (see: *.xpl file with structures)

    ThermalFem2DSolver(const std::string& name="");

    std::string getClassName() const override;

    ~ThermalFem2DSolver();

  protected:

    size_t band;                                ///< Maximum band size
    bool use_full_mesh;                         ///< Should we use full mesh?

    struct ThermalConductivityData: public LazyDataImpl<Tensor2<double>> {
        const ThermalFem2DSolver* solver;
        shared_ptr<const MeshD<2>> dest_mesh;
        InterpolationFlags flags;
        LazyData<double> temps;
        ThermalConductivityData(const ThermalFem2DSolver* solver, const shared_ptr<const MeshD<2>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const override;
        std::size_t size() const override;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) const;

    const LazyData<Vec<2>> getHeatFluxes(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method);

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bvoltage) {
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

    void applyBC(SparseBandMatrix2D& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& bvoltage) {
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

}}} // namespaces

#endif
