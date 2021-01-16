#ifndef PLASK__MODULE_ELECTRICAL_ELECTR2D_H
#define PLASK__MODULE_ELECTRICAL_ELECTR2D_H

#include "common.hpp"
#include "iterative_matrix2d.hpp"

namespace plask { namespace electrical { namespace diffusion {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template <typename GeometryT>
struct PLASK_SOLVER_API ElectricalFem2DSolver : public SolverWithMesh<GeometryT, RectangularMesh<2>> {
  protected:

    /// Details of active region
    struct Active {
        struct Region {
            size_t left, right, bottom, top;
            size_t rowl, rowr;
            bool warn;
            Region()
                : left(0),
                  right(0),
                  bottom(std::numeric_limits<size_t>::max()),
                  top(std::numeric_limits<size_t>::max()),
                  rowl(std::numeric_limits<size_t>::max()),
                  rowr(0),
                  warn(true) {}
        };
        size_t left, right, bottom, top;
        double qwheight;
        std::vector<Box2D> QWs;
        Active() {}
        Active(size_t tot, size_t l, size_t r, size_t b, size_t t, double h)
            : left(l), right(r), bottom(b), top(t), qwheight(h) {}
    };

    size_t band;  ///< Maximum band size

    int loopno;     ///< Number of completed loops
    double toterr;  ///< Maximum estimated error during all iterations (useful for single calculations managed by external python
                    ///< script)

    DataVector<double> conc;        ///< Computed concentration

    std::vector<Active> active;  ///< Active regions information

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix();

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

    /// Get info on active region
    void setActiveRegions();

    void onMeshChange(const typename RectangularMesh<2>::Event& evt) override {
        SolverWithMesh<GeometryT, RectangularMesh<2>>::onMeshChange(evt);
        setActiveRegions();
    }

    void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<GeometryT, RectangularMesh<2>>::onGeometryChange(evt);
        setActiveRegions();
    }

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A,
                   DataVector<double>& B,
                   const BoundaryConditionsWithMesh<RectangularMesh<2>::Boundary, double>& bvoltage,
                   const LazyData<double>& temperature);

    /// Setup matrix
    template <typename MatrixT> MatrixT makeMatrix();

    /// Perform computations for particular matrix type
    template <typename MatrixT> double doCompute(unsigned loops = 1);

    /** Return \c true if the specified point is at QW
     * \param point point to test
     * \returns number of active region + 1 (0 for none)
     */
    size_t isQW(const Vec<2>& point) const {
        size_t no(0);
        auto roles = this->geometry->getRolesAt(point);
        bool qw = false;
        for (auto role : roles) {
            size_t l = 0;
            if (role.substr(0, 6) == "active")
                l = 6;
            else {
                if (role == "QW" || role == "QD") qw = true;
                continue;
            }
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
    size_t isActive(const RectangularMaskedMesh2D::Element& element) const { return isActive(element.getMidpoint()); }

  public:
    double maxerr;  ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>::Boundary, double> voltage_boundary;

    typename ProviderFor<Voltage, GeometryT>::Delegate outVoltage;

    typename ProviderFor<CurrentDensity, GeometryT>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, GeometryT>::Delegate outHeat;

    typename ProviderFor<Conductivity, GeometryT>::Delegate outConductivity;

    ReceiverFor<Temperature, GeometryT> inTemperature;

    Algorithm algorithm;  ///< Factorization algorithm to use

    double itererr;  ///< Allowed residual iteration for iterative method
    size_t iterlim;  ///< Maximum nunber of iterations for iterative method
    size_t logfreq;  ///< Frequency of iteration progress reporting

    /**
     * Run electrical calculations
     * \return max correction of potential against the last call
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
     * Integrate vertical total current flowing vertically through active region.
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

    /// Return the maximum estimated error.
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

    /// Get default juction conductivity [S/m]
    double getCondJunc() const { return junction_conductivity; }
    /// Set default juction conductivity [S/m]
    void setCondJunc(double cond) {
        junction_conductivity.reset(max(junction_conductivity.size(), size_t(1)), cond);
        default_junction_conductivity = cond;
    }
    /// Set default juction conductivity [S/m]
    void setCondJunc(const DataVector<const double>& cond) {
        size_t condsize = 0;
        for (const auto& act : active) condsize += act.right - act.left;
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

    void parseConfiguration(XMLReader& source, Manager& manager);

    ElectricalFem2DSolver(const std::string& name = "");

    ~ElectricalFem2DSolver();

  protected:
    const LazyData<double> getVoltage(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);
};

}}}  // namespace plask::electrical::diffusion

#endif
