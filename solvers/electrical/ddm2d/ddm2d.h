#ifndef PLASK__MODULE_ELECTRICAL_DDM2D_H
#define PLASK__MODULE_ELECTRICAL_DDM2D_H

#include <plask/plask.hpp>
#include <limits>

#include "block_matrix.h"
#include "iterative_matrix.h"
#include "gauss_matrix.h"

namespace plask { namespace solvers { namespace drift_diffusion {

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
struct PLASK_SOLVER_API DriftDiffusionModel2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

    /// Details of active region
    struct Active {
        struct Temp {
            size_t left, right, bottom, top;
            size_t rowl, rowr;
            Temp(): left(0), right(0), bottom(std::numeric_limits<size_t>::max()),
                    top(std::numeric_limits<size_t>::max()),
                    rowl(std::numeric_limits<size_t>::max()), rowr(0) {}
        };
        size_t left, right, bottom, top;
        size_t offset;
        double height;
        Active() {}
        Active(size_t tot, size_t l, size_t r, size_t b, size_t t, double h): left(l), right(r), bottom(b), top(t), offset(tot-l), height(h) {}
    };

  protected:

    int size;                   ///< Number of columns in the main matrix

    bool mRsrh;    ///< SRH recombination is taken into account //LP_09.2015
    bool mRrad;    ///< radiative recombination is taken into account //LP_09.2015
    bool mRaug;    ///< Auger recombination is taken into account //LP_09.2015
    bool mPol;     ///< polarization (GaN is the substrate) //LP_09.2015
    bool mFullIon; ///< dopant ionization = 100% //LP_09.2015

    // scalling parameters
    double mTx;    ///< ambient temperature (K) //LP_09.2015
    double mEx;    ///< energy (eV) //LP_09.2015
    double mNx;    ///< maximal doping concentration (1/cm^3) //LP_09.2015
    double mEpsRx; ///< maximal dielectric constant (-) //LP_09.2015
    double mXx;    ///< sometimes denoted as LD (um) //LP_09.2015
    //double mKx;    ///< thermal conductivity (W/(m*K)) //LP_09.2015
    double mMix;   ///< maximal mobility (cm^2/Vs) //LP_09.2015
    double mRx;    ///< recombination parameter (1/(cm^3*s)) //LP_09.2015
    double mJx;    ///< current density parameter (kA/cm2) //LP_09.2015
    double mtx;    ///< SRH recombination lifetime (s) //LP_09.2015
    double mBx;    ///< radiative recombination coefficient (cm^3/s) //LP_09.2015
    double mCx;    ///< Auger recombination coefficient (cm^6/s) //LP_09.2015
    //double mHx;    ///< heat source (W/(m^3)) //LP_09.2015
    double mPx;    ///< polarization (C/m^2) //LP_09.2015

    double dU;         ///< default voltage step (V)
    double maxDelPsi0; ///< maximal correction for initial potential calculations (V)
    double maxDelPsi;  ///< maximal correction for potential calculations (V)
    double maxDelFn;   ///< maximal correction for quasi-Fermi levels for electrons calculations (eV)
    double maxDelFp;   ///< maximal correction for quasi-Fermi levels for holes calculations (eV)

    std::string stat;  ///< statistics ("MB" - Maxwell-Boltzmann or "FD" - Fermi-Dirac)

    //int loopno;                 ///< Number of completed loops
    //double toterr;              ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)
    //Vec<2,double> maxcur;       ///< Maximum current in the structure

    DataVector<double> dveN;                    ///< Cached electron concentrations (size: elements)
    DataVector<double> dveP;                    ///< Cached hole concentrations (size: elements)
    DataVector<double> dvePsiI;                 ///< Computed initial potentials (size: elements)
    DataVector<double> dvePsi;                  ///< Computed potentials (size: elements)
    DataVector<double> dveFn;                  ///< Computed potentials (size: elements)
    DataVector<double> dveFp;                  ///< Computed potentials (size: elements)
    DataVector<double> dveFnEta;                ///< Computed exponents of quasi-Fermi levels for electrons (size: elements)
    DataVector<double> dveFpKsi;                ///< Computed exponents of quasi-Fermi levels for holes (size: elements)

    DataVector<int> dvnNodeInfo;                ///< Number of elements which have this node (size: nodes)
    DataVector<double> dvnPsi0;                 ///< Computed potential for U=0V (size: nodes)
    DataVector<double> dvnPsi;                  ///< Computed potentials (size: nodes)
    DataVector<double> dvnDeltaPsi;             ///< Computed potentials corrections (size: nodes)
    DataVector<double> dvnFn;                   ///< Computed quasi-Fermi levels for electrons (size: nodes)
    DataVector<double> dvnFp;                   ///< Computed quasi-Fermi levels for holes (size: nodes)
    DataVector<double> dvnFnEta;                ///< Computed exponents of quasi-Fermi levels for electrons (size: nodes)
    DataVector<double> dvnFpKsi;                ///< Computed exponents of quasi-Fermi levels for holes (size: nodes)
    DataVector<double> dvnDeltaFn;              ///< Computed quasi-Fermi levels for electrons corrections (size: nodes)
    DataVector<double> dvnDeltaFp;              ///< Computed quasi-Fermi levels for holes corrections (size: nodes)
    //DataVector<Vec<2,double>> currents;         ///< Computed current densities
    //DataVector<double> heats;                   ///< Computed and cached heat source densities

    void setScaleParam(); ///< set scalling parameters //LP_09.2015
    double findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT, int& loop); ///< find initial potential //LP_09.2015
    double calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT); ///< calculate electron concentration //LP_09.2015
    double calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT); ///< calculate hole concentration //LP_09.2015
    double calcFD12(double iEta); ///< Fermi-Dirac integral of grade 1/2 //LP_09.2015

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix(double& k44, double& k33, double& k22, double& k11,
                               double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
                               double ky, double width, const Vec<2,double>& midpoint);

    void savePsi(); ///< save potentials for all elements to datavector
    void saveFn();  ///< save quasi-Fermi electron level for all elements to datavector
    void saveFp();  ///< save quasi-Fermi electron level for all elements to datavector
    void saveFnEta();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveFpKsi();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveN(); ///< save electron concentrations for all elements to datavector
    void saveP(); ///< save hole concentrations for all elements to datavector
    double addCorr(std::string calctype); ///< add corrections to datavectors
/*
    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();
*/
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
/*
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
*/
    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(std::string calctype, MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(std::string calctype, unsigned loops=1);

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
                catch (boost::bad_lexical_cast) { throw BadInput(this->getId(), "Bad junction number in role '%1%'", role); }
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

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<QuasiFermiElectronLevel, Geometry2DType>::Delegate outQuasiFermiElectronLevel;

    typename ProviderFor<QuasiFermiHoleLevel, Geometry2DType>::Delegate outQuasiFermiHoleLevel;

/*
    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;

    typename ProviderFor<Conductivity, Geometry2DType>::Delegate outConductivity;
*/
    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    ReceiverFor<Wavelength> inWavelength; /// wavelength (for heat generation in the active region) [nm]

    Algorithm algorithm;    ///< Factorization algorithm to use

    double maxerrPsiI;  ///< Maximum estimated error for initial potential during all iterations (useful for single calculations managed by external python script)
    double maxerrPsi0;  ///< Maximum estimated error for potential at U = 0 V during all iterations (useful for single calculations managed by external python script)
    double maxerrPsi;   ///< Maximum estimated error for potential during all iterations (useful for single calculations managed by external python script)
    double maxerrFn;    ///< Maximum estimated error for quasi-Fermi energy level for electrons during all iterations (useful for single calculations managed by external python script)
    double maxerrFp;    ///< Maximum estimated error for quasi-Fermi energy level for holes during all iterations (useful for single calculations managed by external python script)
    size_t iterlimPsiI; ///< Maximum number of iterations for iterative method for initial potential
    size_t iterlimPsi0; ///< Maximum number of iterations for iterative method for potential at U = 0 V
    size_t iterlimPsi;  ///< Maximum number of iterations for iterative method for potential
    size_t iterlimFn;   ///< Maximum number of iterations for iterative method for quasi-Fermi energy level for electrons
    size_t iterlimFp;   ///< Maximum number of iterations for iterative method for quasi-Fermi energy level for holes
    double itererr;         ///< Allowed residual iteration for iterative method
    size_t iterlim;         ///< Maximum number of iterations for iterative method
    size_t logfreq;         ///< Frequency of iteration progress reporting

    /**
     * Calculate initial potential for all elements
     * \return max correction of potential against the last call // TODO
     **/
    int computePsiI(unsigned loops=1);

    /**
     * Set initial potential for all nodes
     * \return max correction of potential against the last call // TODO
     **/
    void setPsiI();

    /**
     * Calculate potential at U=0V for all nodes
     * \return max correction of potential against the last call // TODO
     **/
    int computePsi0(unsigned loops=1);

    /**
     * Run drift_diffusion calculations
     * \return max correction of potential against the last call
     **/
    double compute(std::string calctype, unsigned loops=1);

    /**
     * Increase voltage for p-contact
     * \return nothing
     **/
    void increaseVoltage(double dU, const BoundaryConditionsWithMesh<RectangularMesh<2> ,double>& bvoltage);

    /**
     * Integrate vertical total current at certain level.
     * \param vindex vertical index of the element mesh to perform integration at
     * \param onlyactive if true only current in the active region is considered
     * \return computed total current
     */
    //double integrateCurrent(size_t vindex, bool onlyactive=false);// LP_09.2015

    /**
     * Integrate vertical total current flowing vertically through active region.
     * \param nact number of the active region
     * \return computed total current
     */
    //double getTotalCurrent(size_t nact=0);// LP_09.2015

    /**
     * Compute total electrostatic energy stored in the structure.
     * \return total electrostatic energy [J]
     */
    //double getTotalEnergy();// LP_09.2015

    /**
     * Estimate structure capacitance.
     * \return static structure capacitance [pF]
     */
    //double getCapacitance();// LP_09.2015

    /**
     * Compute total heat generated by the structure in unit time
     * \return total generated heat [mW]
     */
    //double getTotalHeat();// LP_09.2015

    /// Return the maximum estimated error.
    //double getErr() const { return toterr; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager) override; // for solver configuration (see: *.xpl file with structures)

    DriftDiffusionModel2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~DriftDiffusionModel2DSolver();

  protected:
    const LazyData<double> getPotentials(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getQuasiFermiElectronLevels(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getQuasiFermiHoleLevels(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;


    /*
    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);
*/
};

}} //namespaces

} // namespace plask

#endif

