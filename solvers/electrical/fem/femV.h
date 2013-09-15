#ifndef PLASK__MODULE_ELECTRICAL_FEMV_H
#define PLASK__MODULE_ELECTRICAL_FEMV_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "iterative_matrix.h"
#include "gauss_matrix.h"

namespace plask { namespace solvers { namespace electrical {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

/// Type of the returned correction
enum CorrectionType {
    CORRECTION_ABSOLUTE,    ///< absolute correction is used
    CORRECTION_RELATIVE     ///< relative correction is used
};

/// Choice of heat computation method in active region
enum HeatMethod {
    HEAT_JOULES, ///< compute Joules heat using effective conductivity
    HEAT_BANDGAP ///< compute heat based on the size of the band gap
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2DType, RectilinearMesh2D> {

  protected:

    int size;           ///< Number of columns in the main matrix

    double js;          ///< p-n junction parameter [A/m^2]
    double beta;        ///< p-n junction parameter [1/V]
    double pcond;       ///< p-contact electrical conductivity [S/m]
    double ncond;       ///< n-contact electrical conductivity [S/m]

    double corrlim;     ///< Maximum voltage correction accepted as convergence
    int loopno;         ///< Number of completed loops
    double abscorr;     ///< Maximum absolute voltage correction (useful for single calculations managed by external python script)
    double relcorr;     ///< Maximum relative voltage correction (useful for single calculations managed by external python script)
    double dV;          ///< Maximum voltage

    DataVector<double> junction_conductivity;   ///< electrical conductivity for p-n junction in y-direction [S/m]
    double default_junction_conductivity;       ///< default electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> conds;          ///< Cached element conductivities
    DataVector<double> potentials;              ///< Computed potentials
    DataVector<Vec<2,double>> currents;         ///< Computed current densities
    DataVector<double> heats;                   ///< Computed and cached heat source densities

    std::vector<size_t>
        actlo,                  ///< Vertical index of the lower side of the active regions
        acthi;                  ///< Vertical index of the higher side of the active regions
    std::vector<double> actd;   ///< Active regions thickness

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix(double& k44, double& k33, double& k22, double& k11,
                               double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
                               double ky, double width, const Vec<2,double>& midpoint);

    /// Load conductivities
    void loadConductivities();

    /// Save conductivities of active region
    void saveConductivities();

    /// Update stored current densities
    void saveCurrentDensities();

    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();

    /// Matrix solver
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the data
    virtual void onInvalidate() override;

    /// Get info on active region
    void setActiveRegions();

    virtual void onMeshChange(const typename RectilinearMesh2D::Event& evt) override {
        this->invalidate();
        setActiveRegions();
    }

    virtual void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectilinearMesh2D>::onGeometryChange(evt);
        setActiveRegions();
    }

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage);

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage);

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(int loops=1);

  public:

    CorrectionType corrtype; ///< Type of the returned correction

    HeatMethod heatmet;     ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectilinearMesh2D,double> voltage_boundary;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    ReceiverFor<Wavelength> inWavelength; /// wavelength (for heat generation in the active region) [nm]

    Algorithm algorithm;    ///< Factorization algorithm to use

    double itererr;         ///< Allowed residual iteration for iterative method
    size_t iterlim;         ///< Maximum nunber of iterations for iterative method
    size_t logfreq;         ///< Frequency of iteration progress reporting

    /**
     * Run electrical calculations
     * \return max correction of potential against the last call
     **/
    double compute(int loops=1);

    /**
     * Integrate vertical total current at certain level.
     * \param vindex vertical index of the element smesh to perform integration at
     * \return computed total current
     */
    double integrateCurrent(size_t vindex);

    /**
     * Integrate vertical total current flowing vertically through active region
     * \param nact number of the active region
     * \return computed total current
     */
    double getTotalCurrent(size_t nact=0);

    /// \return max absolute correction for potential
    double getMaxAbsVCorr() const { return abscorr; } // result in [K]

    /// \return get max relative correction for potential
    double getMaxRelVCorr() const { return relcorr; } // result in [%]

    double getVCorrLim() const { return corrlim; }
    void setVCorrLim(double lim) { corrlim = lim; }

    double getBeta() const { return beta; }
    void setBeta(double beta)  {
        this->beta = beta;
        this->invalidate();
    }

    double getJs() const { return js; }
    void setJs(double js)  {
        this->js = js;
        this->invalidate();
    }

    double getCondPcontact() const { return pcond; }
    void setCondPcontact(double cond)  { pcond = cond; }

    double getCondNcontact() const { return ncond; }
    void setCondNcontact(double cond)  { ncond = cond; }

    DataVector<const double> getCondJunc() const { return junction_conductivity; }
    void setCondJunc(double cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = cond;
    }
    void setCondJunc(const DataVector<const double>& cond)  {
        if (!this->mesh || cond.size() != (this->mesh->axis0.size()-1) * getActNo())
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        junction_conductivity = cond.claim();
    }

    double getActLo(size_t n) const { return actlo[n]; }
    double getActHi(size_t n) const { return acthi[n]; }
    size_t getActNo() const { return actd.size(); }

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodElectrical2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~FiniteElementMethodElectrical2DSolver();

  protected:

    DataVector<const double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);

    DataVector<const Vec<2>> getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);

};

}} //namespaces

} // namespace plask

#endif

