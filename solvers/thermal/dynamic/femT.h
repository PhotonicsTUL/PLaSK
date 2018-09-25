#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "gauss_matrix.h"
#include "algorithm.h"

namespace plask { namespace thermal { namespace dynamic {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API FiniteElementMethodDynamicThermal2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    std::size_t size;   ///< Number of columns in the main matrix
    double maxT;        ///< Maximum temperature recorded

    DataVector<double> temperatures;            ///< Computed temperatures

    DataVector<double> thickness;               ///< Thicknesses of the layers

    DataVector<Vec<2,double>> fluxes;      ///< Computed (only when needed) heat fluxes on our own mesh

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, MatrixT& B, DataVector<double>& F,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary,double>& btemperature
                  );

    /// Update stored temperatures and calculate corrections
    double saveTemperatures(DataVector<double>& T);

    /// Create 2D-vector with calculated heat fluxes
    void saveHeatFluxes(); // [W/m^2]

    /// Matrix preparation
    void prepareMatrix(DpbMatrix& A);

    /// Matrix solver
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix preparation
    void prepareMatrix(DgbMatrix& A);

    /// Matrix solver
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the data
    virtual void onInvalidate() override;

  public:

    // Boundary conditions
    BoundaryConditions<RectangularMesh<2>::Boundary,double> temperature_boundary;      ///< Boundary condition of constant temperature [K]

    typename ProviderFor<Temperature, Geometry2DType>::Delegate outTemperature;

    typename ProviderFor<HeatFlux, Geometry2DType>::Delegate outHeatFlux;

    typename ProviderFor<ThermalConductivity, Geometry2DType>::Delegate outThermalConductivity;

    ReceiverFor<Heat, Geometry2DType> inHeat;

    Algorithm algorithm;   ///< Factorization algorithm to use

    double inittemp;       ///< Initial temperature
    double methodparam;   ///< Initial parameter determining the calculation method (0.5 - Crank-Nicolson, 0 - explicit, 1 - implicit)
    double timestep;       ///< Time step in nanoseconds
    double elapstime;    ///< Calculations elapsed time
    bool lumping;          ///< Wheter use lumping for matrices?
    size_t rebuildfreq;    ///< Frequency of mass matrix rebuilding
    size_t logfreq;        ///< Frequency of iteration progress reporting

    /**
     * Run temperature calculations
     * \return max correction of temperature against the last call
     **/
    double compute(double time);

    /// Get calculations elapsed time
    double getElapsTime() const { return elapstime; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager) override; // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodDynamicThermal2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~FiniteElementMethodDynamicThermal2DSolver();

  protected:

    struct ThermalConductivityData: public LazyDataImpl<Tensor2<double>> {
        const FiniteElementMethodDynamicThermal2DSolver* solver;
        shared_ptr<const MeshD<2>> dest_mesh;
        InterpolationFlags flags;
        LazyData<double> temps;
        ThermalConductivityData(const FiniteElementMethodDynamicThermal2DSolver* solver, const shared_ptr<const MeshD<2>>& dst_mesh);
        Tensor2<double> at(std::size_t i) const override;
        std::size_t size() const override;
    };

    const LazyData<double> getTemperatures(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method) const;

    const LazyData<Vec<2>> getHeatFluxes(const shared_ptr<const MeshD<2>>& dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getThermalConductivity(const shared_ptr<const MeshD<2>>& dst_mesh, InterpolationMethod method);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(double time);

};

}} //namespaces

} // namespace plask

#endif

