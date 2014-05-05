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

/// Choice of heat computation method in active region
enum HeatMethod {
    HEAT_JOULES, ///< compute Joules heat using effective conductivity
    HEAT_BANDGAP ///< compute heat based on the size of the band gap
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    int size;               ///< Number of columns in the main matrix

    double js;              ///< p-n junction parameter [A/m^2]
    double beta;            ///< p-n junction parameter [1/V]
    double pcond;           ///< p-contact electrical conductivity [S/m]
    double ncond;           ///< n-contact electrical conductivity [S/m]

    int loopno;             ///< Number of completed loops
    double toterr;          ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)
    Vec<2,double> maxcur;   ///< Maximum current in the structure

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

    virtual void onMeshChange(const typename RectangularMesh<2>::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onMeshChange(evt);
        setActiveRegions();
    }

    virtual void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onGeometryChange(evt);
        setActiveRegions();
    }

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(unsigned loops=1);

    /// Return \c true if the specified point is at junction
    bool isActive(const Vec<2>& point) const {
        auto roles = this->geometry->getRolesAt(point);
        return roles.find("active") != roles.end() || roles.find("junction") != roles.end();
    }

    /// Return \c true if the specified element is a junction
    bool isActive(const RectangularMesh<2>::Element& element) const {
           return isActive(element.getMidpoint());
    }

  public:

    double maxerr;              ///< Maximum relative current density correction accepted as convergence

    HeatMethod heatmet;         ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2> ,double> voltage_boundary;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;

    typename ProviderFor<ElectricalConductivity, Geometry2DType>::Delegate outConductivity;

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
    double compute(unsigned loops=1);

    /**
     * Integrate vertical total current at certain level.
     * \param vindex vertical index of the element smesh to perform integration at
     * \return computed total current
     */
    double integrateCurrent(size_t vindex);

    /**
     * Integrate vertical total current flowing vertically through active region.
     * \param nact number of the active region
     * \return computed total current
     */
    double getTotalCurrent(size_t nact=0);

    /**
     * Compute total electrostatic energy stored in the structure.
     * \return total electrostatic energy [J]
     */
    double getTotalEnergy();
    
    /**
     * Estimate structure capacitance.
     * \return static structure capacitance [pF]
     */
    double getCapacitance();

    /// Return the maximum estimated error.
    double getErr() const { return toterr; }

    /// Return beta.
    double getBeta() const { return beta; }
    /// Set new beta and invalidate the solver.
    void setBeta(double beta)  {
        this->beta = beta;
        this->invalidate();
    }

    /// Get junction thermal voltage.
    double getVt() const { return 1. / beta; }
    /// Set new junction thermal voltage and invalidate the solver.
    void setVt(double Vt) {
        this->beta = 1. / Vt;
        this->invalidate();
    }

    /// Return js
    double getJs() const { return js; }
    /// Set new js and invalidate the solver
    void setJs(double js)  {
        this->js = js;
        this->invalidate();
    }

    double getCondPcontact() const { return pcond; }
    void setCondPcontact(double cond)  {
        pcond = cond;
        this->invalidate();
    }

    double getCondNcontact() const { return ncond; }
    void setCondNcontact(double cond)  {
        ncond = cond;
        this->invalidate();
    }

    DataVector<const double> getCondJunc() const { return junction_conductivity; }
    void setCondJunc(double cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = cond;
    }
    void setCondJunc(const DataVector<const double>& cond)  {
        if (!this->mesh || cond.size() != (this->mesh->axis0->size()-1) * getActNo())
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

    DataVector<const Tensor2<double>> getConductivity(const MeshD<2>& dst_mesh, InterpolationMethod method);
};

}} //namespaces

} // namespace plask

#endif

