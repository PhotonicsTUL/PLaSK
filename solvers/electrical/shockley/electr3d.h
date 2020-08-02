#ifndef PLASK__MODULE_ELECTRICAL_ELECTR3D_H
#define PLASK__MODULE_ELECTRICAL_ELECTR3D_H

#include "common.h"
#include "iterative_matrix3d.h"

namespace plask { namespace electrical { namespace shockley {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
struct PLASK_SOLVER_API ElectricalFem3DSolver : public SolverWithMesh<Geometry3D, RectangularMesh<3>> {
  protected:
    plask::shared_ptr<RectangularMaskedMesh3D> maskedMesh = plask::make_shared<RectangularMaskedMesh3D>();

    /// Details of active region
    struct Active {
        struct Region {
            size_t bottom, top, left, right, back, front, lon, tra;
            bool warn;
            Region() {}
            Region(size_t b, size_t t, size_t x, size_t y)
                : bottom(b),
                  top(t),
                  left(std::numeric_limits<size_t>::max()),
                  right(0),
                  back(std::numeric_limits<size_t>::max()),
                  front(0),
                  lon(x),
                  tra(y),
                  warn(true) {}
        };
        size_t bottom, top, left, right, back, front;
        size_t ld;
        ptrdiff_t offset;
        double height;
        Active() {}
        Active(size_t tot, size_t v0, size_t v1, size_t t0, size_t t1, size_t l0, size_t l1, double h)
            : bottom(v0),
              top(v1),
              left(t0),
              right(t1),
              back(l0),
              front(l1),
              ld(l1 - l0),
              offset(tot - (l1 - l0) * t0 - l0),
              height(h) {}
        Active(size_t tot, Region r, double h)
            : bottom(r.bottom),
              top(r.top),
              left(r.left),
              right(r.right),
              back(r.back),
              front(r.front),
              ld(r.front - r.back),
              offset(tot - (r.front - r.back) * r.left - r.back),
              height(h) {}
    };

    size_t band;  ///< Maximum band size

    double pcond;  ///< p-contact electrical conductivity [S/m]
    double ncond;  ///< n-contact electrical conductivity [S/m]

    int loopno;     ///< Number of completed loops
    double toterr;  ///< Maximum estimated error during all iterations (useful for single calculations managed by external python
                    ///< script)

    DataVector<double> junction_conductivity;  ///< electrical conductivity for p-n junction in y-direction [S/m]
    double default_junction_conductivity;      ///< default electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> conds;  ///< Cached element conductivities

    DataVector<double> potential;        ///< Computed potentials
    DataVector<Vec<3, double>> current;  ///< Computed current densities
    DataVector<double> heat;             ///< Computed and cached heat source densities

    std::vector<Active> active;  ///< Active regions information

    bool use_full_mesh;  ///< Should we use full mesh?

    /**
     * Set stiffness matrix and load vector
     * \param[out] A matrix to fill-in
     * \param[out] B load vector
     * \param bvoltage boundary conditions: constant voltage
     **/
    template <typename MatrixT>
    void setMatrix(MatrixT& A,
                   DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary, double>& bvoltage,
                   const LazyData<double>& temperature);

    /// Apply boundary conditions of the first kind
    template <typename MatrixT>
    void applyBC(MatrixT& A,
                 DataVector<double>& B,
                 const BoundaryConditionsWithMesh<RectangularMesh<3>::Boundary, double>& bvoltage);

    /** Compute voltage drop of the active region
     *  \param n active region number
     *  \param jy vertical current [kA/cm²]
     *  \param T temperature [K]
     */
    virtual double activeVoltage(size_t n, double jy, double T) = 0;

    /** Load conductivities
     *  \return current temperature
     */
    LazyData<double> loadConductivity();

    /// Save conductivities of active region
    void saveConductivity();

    /// Create 3D-vector with calculated heat density
    void saveHeatDensity();

    /// Matrix solver for the block cholesky algorithm
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver for the block gauss algorithm
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver for the iterative algorithm
    void solveMatrix(SparseBandMatrix3D& A, DataVector<double>& B);

    /// Initialize the solver
    void onInitialize() override;

    /// Invalidate the data
    void onInvalidate() override;

    void onMeshChange(const typename RectangularMesh<3>::Event& evt) override {
        SolverWithMesh<Geometry3D, RectangularMesh<3>>::onMeshChange(evt);
        setActiveRegions();
    }

    void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry3D, RectangularMesh<3>>::onGeometryChange(evt);
        setActiveRegions();
    }

    /// Get info on active region
    void setActiveRegions();

    /// Setup matrix
    template <typename MatrixT> MatrixT makeMatrix();

    /// Perform computations for particular matrix type
    template <typename MatrixT> double doCompute(unsigned loops = 1);

    /// Return \c true if the specified point is at junction
    size_t isActive(const Vec<3>& point) const {
        size_t no(0);
        auto roles = this->geometry->getRolesAt(point);
        for (auto role : roles) {
            size_t l = 0;
            if (role.substr(0, 6) == "active")
                l = 6;
            else if (role.substr(0, 8) == "junction")
                l = 8;
            else
                continue;
            if (no != 0) throw BadInput(this->getId(), "Multiple 'active'/'junction' roles specified");
            if (role.size() == l)
                no = 1;
            else {
                try {
                    no = boost::lexical_cast<size_t>(role.substr(l)) + 1;
                } catch (boost::bad_lexical_cast&) {
                    throw BadInput(this->getId(), "Bad junction number in role '{0}'", role);
                }
            }
        }
        return no;
    }

    /// Return \c true if the specified element is a junction
    size_t isActive(const RectangularMesh<3>::Element& element) const { return isActive(element.getMidpoint()); }

    /// Return \c true if the specified element is a junction
    size_t isActive(const RectangularMaskedMesh<3>::Element& element) const { return isActive(element.getMidpoint()); }

  public:
    Algorithm algorithm;  ///< Factorization algorithm to use

    double maxerr;          ///< Maximum relative current density correction accepted as convergence
    Vec<3, double> maxcur;  ///< Maximum current in the structure

    double itererr;  ///< Allowed residual iteration for iterative method
    size_t iterlim;  ///< Maximum number of iterations for iterative method

    size_t logfreq;  ///< Frequency of iteration progress reporting

    // Boundary conditions
    BoundaryConditions<RectangularMesh<3>::Boundary, double> voltage_boundary;  ///< Boundary condition of constant voltage [K]

    typename ProviderFor<Voltage, Geometry3D>::Delegate outVoltage;

    typename ProviderFor<CurrentDensity, Geometry3D>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry3D>::Delegate outHeat;

    typename ProviderFor<Conductivity, Geometry3D>::Delegate outConductivity;

    ReceiverFor<Temperature, Geometry3D> inTemperature;

    ElectricalFem3DSolver(const std::string& name = "");

    void parseConfiguration(XMLReader& source, Manager& manager);

    ~ElectricalFem3DSolver();

    /**
     * Run voltage calculations
     * \param loops maximum number of loops to run
     * \return max correction of voltage against the last call
     **/
    double compute(unsigned loops = 1);

    /**
     * Integrate vertical total current at certain level.
     * \param vindex vertical index of the element mesh to perform integration at
     * \param onlyactive if true only current in the active region is considered
     * \return computed total current
     */
    double integrateCurrent(size_t vindex, bool onlyactive = false);
    /**
     * Integrate vertical total current flowing vertically through active region
     * \param nact number of the active region
     * \return computed total current
     */
    double getTotalCurrent(size_t nact = 0);

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

    /**
     * Compute total heat generated by the structure in unit time
     * \return total generated heat [mW]
     */
    double getTotalHeat();

    /// Return the maximum estimated error
    double getErr() const { return toterr; }

    /// Get p-contact layer conductivity [S/m]
    double getCondPcontact() const { return pcond; }
    /// Set p-contact layer conductivity [S/m]
    void setCondPcontact(double cond) {
        pcond = cond;
        this->invalidate();
    }

    /// Get n-contact layer conductivity [S/m]
    double getCondNcontact() const { return ncond; }
    /// Set n-contact layer conductivity [S/m]
    void setCondNcontact(double cond) {
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
    void setCondJunc(const DataVector<const double>& cond) {
        size_t condsize = 0;
        for (const auto& act : active) condsize += (act.right - act.left) * act.ld;
        condsize = max(condsize, size_t(1));
        if (!this->mesh || cond.size() != condsize)
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        junction_conductivity = cond.claim();
    }

    /// Are we using full mesh?
    bool usingFullMesh() const { return use_full_mesh; }
    /// Set whether we should use full mesh
    void useFullMesh(bool val) {
        use_full_mesh = val;
        setActiveRegions();
    }

  protected:
    const LazyData<double> getVoltage(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method) const;

    const LazyData<Vec<3>> getCurrentDensity(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method);

    const LazyData<double> getHeatDensity(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<3>> dest_mesh, InterpolationMethod method);
};

}}}  // namespace plask::electrical::shockley

#endif
