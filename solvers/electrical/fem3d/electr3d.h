#ifndef PLASK__MODULE_THERMAL_FEMT_H
#define PLASK__MODULE_THERMAL_FEMT_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "iterative_matrix.h"

namespace plask { namespace solvers { namespace electrical3d {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< block algorithm (thrice faster, however a little prone to failures)
    ALGORITHM_ITERATIVE ///< iterative algorithm using preconditioned conjugate gradient method
};

/// Choice of heat computation method in active region
enum HeatMethod {
    HEAT_JOULES, ///< compute Joules heat using effective conductivity
    HEAT_BANDGAP ///< compute heat based on the size of the band gap
};


/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct FiniteElementMethodElectrical3DSolver: public SolverWithMesh<Geometry3D,RectilinearMesh3D> {

  protected:

    double js;                                  ///< p-n junction parameter [A/m^2]
    double beta;                                ///< p-n junction parameter [1/V]
    double pcond;                               ///< p-contact electrical conductivity [S/m]
    double ncond;                               ///< n-contact electrical conductivity [S/m]

    Algorithm algorithm;                        ///< Factorization algorithm to use

    int loopno;                                 ///< Number of completed loops
    double toterr;                              ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)

    DataVector<double> junction_conductivity;   ///< electrical conductivity for p-n junction in y-direction [S/m]
    double default_junction_conductivity;       ///< default electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> conds;          ///< Cached element conductivities

    DataVector<double> potential;               ///< Computed potentials
    DataVector<Vec<3,double>> current;          ///< Computed current densities
    DataVector<double> heat;                    ///< Computed and cached heat source densities

    std::vector<size_t>
        actlo,                                  ///< Vertical index of the lower side of the active regions
        acthi;                                  ///< Vertical index of the higher side of the active regions
    std::vector<double> actd;                   ///< Active regions thickness


    /**
     * Set stiffness matrix and load vector
     * \param[out] A matrix to fill-in
     * \param[out] B load vector
     * \param bvoltage boundary conditions: constant voltage
     **/
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bvoltage);

    /**
     * Apply boundary conditions of the first kind
     */
    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh3D,double>& bvoltage);

    /// Load conductivities
    void loadConductivity();

    /// Save conductivities of active region
    void saveConductivity();

    /// Create 3D-vector with calculated heat density
    void saveHeatDensity();

    /// Matrix solver for block algorithm
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver for iterative algorithm
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    /// Initialize the solver
    virtual void onInitialize();

    /// Invalidate the data
    virtual void onInvalidate();

    virtual void onMeshChange(const typename RectilinearMesh3D::Event& evt) override {
        SolverWithMesh<Geometry3D,RectilinearMesh3D>::onMeshChange(evt);
        setActiveRegions();
    }

    virtual void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry3D,RectilinearMesh3D>::onGeometryChange(evt);
        setActiveRegions();
    }

    /// Get info on active region
    void setActiveRegions();

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(unsigned loops=1);

    /// Return \c true if the specified point is at junction
    bool isActive(const Vec<3>& point) const {
        auto roles = geometry->getRolesAt(point);
        return roles.find("active") != roles.end() || roles.find("junction") != roles.end();
    }

    /// Return \c true if the specified element is a junction
    bool isActive(const RectilinearMesh3D::Element& element) const {
           return isActive(element.getMidpoint());
    }

  public:

    double maxerr;              ///< Maximum relative current density correction accepted as convergence
    Vec<3,double> maxcur;       ///< Maximum current in the structure

    HeatMethod heatmet;         ///< Method of heat computation


    double itererr;             ///< Allowed residual iteration for iterative method
    size_t iterlim;             ///< Maximum number of iterations for iterative method

    size_t logfreq;             ///< Frequency of iteration progress reporting

    // Boundary conditions
    BoundaryConditions<RectilinearMesh3D,double> voltage_boundary;      ///< Boundary condition of constant voltage [K]

    typename ProviderFor<Potential,Geometry3D>::Delegate outPotential;

    typename ProviderFor<CurrentDensity,Geometry3D>::Delegate outCurrentDensity;

    typename ProviderFor<Heat,Geometry3D>::Delegate outHeat;

    typename ProviderFor<ElectricalConductivity,Geometry3D>::Delegate outConductivity;

    ReceiverFor<Temperature,Geometry3D> inTemperature;

    ReceiverFor<Wavelength> inWavelength; /// wavelength (for heat generation in the active region) [nm]


    FiniteElementMethodElectrical3DSolver(const std::string& name="");

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    virtual std::string getClassName() const { return "electrical.Shockley3D"; }

    ~FiniteElementMethodElectrical3DSolver();

    /**
     * Run voltage calculations
     * \param loops maximum number of loops to run
     * \return max correction of voltage against the last call
     **/
    double compute(unsigned loops=1);

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

    /// Return maximum estimated error
    double getErr() const { return toterr; }

    /// \return current algorithm
    Algorithm getAlgorithm() const { return algorithm; }

    /// Set algorithm
    void setAlgorithm(Algorithm alg) {
        algorithm = alg;
    }

    /// Get \f$ \beta \f$ [1/V]
    double getBeta() const { return beta; }
    /// Set \f$ \beta \f$ [1/V]
    void setBeta(double beta)  {
        this->beta = beta;
        this->invalidate();
    }

    /// Get junction thermal voltage \f$ V_t \f$
    double getVt() const { return 1. / beta; }
    /// Set new junction thermal voltage \f$ V_t \f$ and invalidate the solver
    void setVt(double Vt) {
        this->beta = 1. / Vt;
        this->invalidate();
    }

    /// Get \f$ j_s \f$ [A/m²]
    double getJs() const { return js; }
    /// Set \f$ j_s \f$ [A/m²]
    void setJs(double js)  {
        this->js = js;
        this->invalidate();
    }

    /// Get p-contact layer conductivity [S/m]
    double getPcond() const { return pcond; }
    /// Set p-contact layer conductivity [S/m]
    void setPcond(double cond)  {
        pcond = cond;
        this->invalidate();
    }

    /// Get n-contact layer conductivity [S/m]
    double getNcond() const { return ncond; }
    /// Set n-contact layer conductivity [S/m]
    void setNcond(double cond)  {
        ncond = cond;
        this->invalidate();
    }

    /// Get data with junction effective conductivity
    DataVector<const double> getCondJunc() const { return junction_conductivity; }
    /// Set junction effective conductivity to the single value
    void setCondJunc(double cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = cond;
    }
    /// Set junction effective conductivity to previously read data
    void setCondJunc(const DataVector<const double>& cond)  {
        if (!this->mesh || cond.size() != (this->mesh->axis0.size()-1) * (this->mesh->axis1.size()-1) * getActNo())
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        junction_conductivity = cond.claim();
    }

    double getActLo(size_t n) const { return actlo[n]; }
    double getActHi(size_t n) const { return acthi[n]; }
    size_t getActNo() const { return actd.size(); }

  protected:

    DataVector<const double> getPotential(const MeshD<3>& dst_mesh, InterpolationMethod method) const;

    DataVector<const Vec<3>> getCurrentDensity(const MeshD<3>& dst_mesh, InterpolationMethod method);

    DataVector<const double> getHeatDensity(const MeshD<3>& dst_mesh, InterpolationMethod method);

    DataVector<const Tensor2<double>> getConductivity(const MeshD<3>& dst_mesh, InterpolationMethod method);
};

}}} //namespaces

#endif

