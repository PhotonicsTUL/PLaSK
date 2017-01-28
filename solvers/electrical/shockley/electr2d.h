#ifndef PLASK__MODULE_ELECTRICAL_ELECTR2D_H
#define PLASK__MODULE_ELECTRICAL_ELECTR2D_H

#include "common.h"
#include "iterative_matrix2d.h"
#include <limits>

namespace plask { namespace electrical { namespace shockley {

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API FiniteElementMethodElectrical2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    /// Details of active region
    struct Active {
        struct Region {
            size_t left, right, bottom, top;
            size_t rowl, rowr;
            bool warn;
            Region(): left(0), right(0), bottom(std::numeric_limits<size_t>::max()),
                      top(std::numeric_limits<size_t>::max()),
                      rowl(std::numeric_limits<size_t>::max()), rowr(0), warn(true) {}
        };
        size_t left, right, bottom, top;
        ptrdiff_t offset;
        double height;
        Active() {}
        Active(size_t tot, size_t l, size_t r, size_t b, size_t t, double h): left(l), right(r), bottom(b), top(t), offset(tot-l), height(h) {}
    };

    int size;                   ///< Number of columns in the main matrix

    std::vector<double> js;     ///< p-n junction parameter [A/m^2]
    std::vector<double> beta;   ///< p-n junction parameter [1/V]
    double pcond;               ///< p-contact electrical conductivity [S/m]
    double ncond;               ///< n-contact electrical conductivity [S/m]

    int loopno;                 ///< Number of completed loops
    double toterr;              ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)
    Vec<2,double> maxcur;       ///< Maximum current in the structure

    DataVector<double> junction_conductivity;   ///< electrical conductivity for p-n junction in y-direction [S/m]
    double default_junction_conductivity;       ///< default electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> conds;          ///< Cached element conductivities
    DataVector<double> potentials;              ///< Computed potentials
    DataVector<Vec<2,double>> currents;         ///< Computed current densities
    DataVector<double> heats;                   ///< Computed and cached heat source densities

    std::vector<Active> active;                 ///< Active regions information

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
    void solveMatrix(SparseBandMatrix2D& A, DataVector<double>& B);

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
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    void applyBC(SparseBandMatrix2D& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(unsigned loops=1);

    /** Return \c true if the specified point is at junction
     * \param point point to test
     * \returns number of active region + 1 (0 for none)
     */
    size_t isActive(const Vec<2>& point) const {
        size_t no(0);
        auto roles = this->geometry->getRolesAt(point);
        for (auto role: roles) {
            size_t l = 0;
            if (role.substr(0,6) == "active") l = 6;
            else if (role.substr(0,8)  == "junction") l = 8;
            else continue;
            if (no != 0) throw BadInput(this->getId(), "Multiple 'active'/'junction' roles specified");
            if (role.size() == l)
                no = 1;
            else {
                try { no = boost::lexical_cast<size_t>(role.substr(l)) + 1; }
                catch (boost::bad_lexical_cast) { throw BadInput(this->getId(), "Bad junction number in role '{0}'", role); }
            }
        }
        return no;
    }

    /// Return \c true if the specified element is a junction
    size_t isActive(const RectangularMesh<2>::Element& element) const { return isActive(element.getMidpoint()); }

  public:

    double maxerr;              ///< Maximum relative current density correction accepted as convergence

    HeatMethod heatmet;         ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>,double> voltage_boundary;

    typename ProviderFor<Voltage,Geometry2DType>::Delegate outVoltage;

    typename ProviderFor<CurrentDensity,Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat,Geometry2DType>::Delegate outHeat;

    typename ProviderFor<Conductivity,Geometry2DType>::Delegate outConductivity;

    ReceiverFor<Temperature,Geometry2DType> inTemperature;

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
     * \param vindex vertical index of the element mesh to perform integration at
     * \param onlyactive if true only current in the active region is considered
     * \return computed total current
     */
    double integrateCurrent(size_t vindex, bool onlyactive=false);

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

    /**
     * Compute total heat generated by the structure in unit time
     * \return total generated heat [mW]
     */
    double getTotalHeat();

    /// Return the maximum estimated error.
    double getErr() const { return toterr; }

    /// Return beta.
    double getBeta(size_t n) const {
        if (beta.size() <= n) throw Exception("{0}: no beta given for junction {1}", this->getId(), n);
        return beta[n];
    }
    /// Set new beta and invalidate the solver.
    void setBeta(size_t n, double beta)  {
        if (this->beta.size() <= n) {
            this->beta.reserve(n+1); for (size_t s = this->beta.size(); s <= n; ++s) this->beta.push_back(NAN);
        }
        this->beta[n] = beta;
        this->invalidate();
    }

    /// Get junction thermal voltage.
    double getVt(size_t n) const {
        if (beta.size() <= n) throw Exception("{0}: no Vt given for junction {1}", this->getId(), n);
        return 1. / beta[n];
    }
    /// Set new junction thermal voltage and invalidate the solver.
    void setVt(size_t n, double Vt) {
        if (beta.size() <= n) {
            beta.reserve(n+1); for (size_t s = beta.size(); s <= n; ++s) beta.push_back(NAN);
        }
        this->beta[n] = 1. / Vt;
        this->invalidate();
    }

    /// Return js
    double getJs(size_t n) const {
        if (js.size() <= n) throw Exception("{0}: no js given for junction {1}", this->getId(), n);
        return js[n];
    }
    /// Set new js and invalidate the solver
    void setJs(size_t n, double js)  {
        if (this->js.size() <= n) {
            this->js.reserve(n+1); for (size_t s = this->js.size(); s <= n; ++s) this->js.push_back(1.);
        }
        this->js[n] = js;
        this->invalidate();
    }

    /// Get p-contact layer conductivity [S/m]
    double getCondPcontact() const { return pcond; }
    /// Set p-contact layer conductivity [S/m]
    void setCondPcontact(double cond)  {
        pcond = cond;
        this->invalidate();
    }

    /// Get n-contact layer conductivity [S/m]
    double getCondNcontact() const { return ncond; }
    /// Set n-contact layer conductivity [S/m]
    void setCondNcontact(double cond)  {
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
    void setCondJunc(const DataVector<const double>& cond)  {
        size_t condsize = 0;
        for (const auto& act: active) condsize += act.right - act.left;
        condsize = max(condsize, size_t(1));
        if (!this->mesh || cond.size() != condsize)
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        junction_conductivity = cond.claim();
    }

    virtual void loadConfiguration(XMLReader& source, Manager& manager) override; // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodElectrical2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~FiniteElementMethodElectrical2DSolver();

  protected:

    const LazyData<double> getVoltage(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);
};

}}} //namespaces

#endif

