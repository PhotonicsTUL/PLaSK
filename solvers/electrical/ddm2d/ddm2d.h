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

/// Carrier statistics types
enum Statistics {
    STAT_MB,            ///< Maxwell-Boltzmann
    STAT_FD             ///< Fermi-Dirac
};

/// Type of calculations passed to some functions
enum CalcType {
    CALC_PSI0,          ///< Initial potential
    CALC_PSI,           ///< Potential
    CALC_FN,            ///< Quasi-Fermi level for electrons
    CALC_FP             ///< Quasi-Fermi level for holes
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

    bool mRsrh;    ///< SRH recombination is taken into account
    bool mRrad;    ///< radiative recombination is taken into account
    bool mRaug;    ///< Auger recombination is taken into account
    bool mPol;     ///< polarization (GaN is the substrate)
    bool mFullIon; ///< dopant ionization = 100%

    // scalling parameters
    double mTx;    ///< ambient temperature (K)
    double mEx;    ///< energy (eV)
    double mNx;    ///< maximal doping concentration (1/cm^3)
    double mEpsRx; ///< maximal dielectric constant (-)
    double mXx;    ///< sometimes denoted as LD (um)
    //double mKx;    ///< thermal conductivity (W/(m*K))
    double mMix;   ///< maximal mobility (cm^2/Vs)
    double mRx;    ///< recombination parameter (1/(cm^3*s))
    double mJx;    ///< current density parameter (kA/cm2)
    double mtx;    ///< SRH recombination lifetime (s)
    double mBx;    ///< radiative recombination coefficient (cm^3/s)
    double mCx;    ///< Auger recombination coefficient (cm^6/s)
    //double mHx;    ///< heat source (W/(m^3))
    double mPx;    ///< polarization (C/m^2)

    double dU;         ///< default voltage step (V)
    double maxDelPsi0; ///< maximal correction for initial potential calculations (V)
    double maxDelPsi;  ///< maximal correction for potential calculations (V)
    double maxDelFn;   ///< maximal correction for quasi-Fermi levels for electrons calculations (eV)
    double maxDelFp;   ///< maximal correction for quasi-Fermi levels for holes calculations (eV)

    Statistics stat;  ///< carriers statistics

    //int loopno;                 ///< Number of completed loops
    //double toterr;              ///< Maximum estimated error during all iterations (useful for single calculations managed by external python script)
    //Vec<2,double> maxcur;       ///< Maximum current in the structure

    DataVector<double> dveN;                    ///< Cached electron concentrations (size: elements)
    DataVector<double> dveP;                    ///< Cached hole concentrations (size: elements)
    DataVector<double> dvePsi;                  ///< Computed potentials (size: elements)
    DataVector<double> dveFnEta;                ///< Computed exponents of quasi-Fermi levels for electrons (size: elements)
    DataVector<double> dveFpKsi;                ///< Computed exponents of quasi-Fermi levels for holes (size: elements)

    DataVector<double> dvnPsi0;                 ///< Computed potential for U=0V (size: nodes)
    DataVector<double> dvnPsi;                  ///< Computed potentials (size: nodes)
    DataVector<double> dvnFnEta;                ///< Computed exponents of quasi-Fermi levels for electrons (size: nodes)
    DataVector<double> dvnFpKsi;                ///< Computed exponents of quasi-Fermi levels for holes (size: nodes)
    //DataVector<Vec<2,double>> currents;         ///< Computed current densities
    //DataVector<double> heats;                   ///< Computed and cached heat source densities

    bool needPsi0;                             ///< Flag indicating if we need to compute initial potential;

    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the data
    virtual void onInvalidate() override;

    /**
     * Calculate initial potential for all elements
     */
    void computePsiI();

  private:

    /// Slot called when gain has changed
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        needPsi0 = true;
    }

    /// Find initial potential
    double findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT, int& loop) const;

    /// Calculate electron concentration
    double calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT) const { 
        double yn;
        switch (stat) {
            case STAT_MB: yn = 1.; break;
            case STAT_FD: yn = calcFD12(log(iFnEta) + iPsi - iEc0) / (iFnEta * exp(iPsi-iEc0)); break;
        }
        return iNc * iFnEta * yn * exp(iPsi-iEc0);
    }

    /// Calculate hole concentration
    double calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT) const {
        double yp;
        switch (stat) {
            case STAT_MB: yp = 1.; break;
            case STAT_FD: yp = calcFD12(log(iFpKsi) - iPsi + iEv0) / (iFpKsi * exp(-iPsi+iEv0)); break;
        }
        return iNv * iFpKsi * yp * exp(iEv0-iPsi);
    }

    void divideByElements(DataVector<double>& values) {
        size_t majs = this->mesh->majorAxis()->size(), mins = this->mesh->minorAxis()->size();
        if (mins == 0 || majs == 0) return;
        for (size_t j = 1, jend = mins-1; j < jend; ++j) values[j] *= 0.5;
        for (size_t i = 1, iend = majs-1; i < iend; ++i) {
            values[mins*i] *= 0.5;
            for (size_t j = 1, jend = mins-1; j < jend; ++j) values[mins*i+j] *= 0.25;
            values[mins*(i+1)-1] *= 0.5;
        }
        for (size_t j = mins*(majs-1)+1, jend = this->mesh->size()-1; j < jend; ++j) values[j] *= 0.5;
    }

    /// Fermi-Dirac integral of grade 1/2
    double calcFD12(double iEta) const {
        double tKsi = pow(iEta,4.) + 33.6*iEta*(1.-0.68*exp(-0.17*(iEta+1.)*(iEta+1.))) + 50.;
        return 0.5*sqrt(M_PI) / (0.75*sqrt(M_PI)*pow(tKsi,-0.375) + exp(-iEta));
    }

    void savePsi0(); ///< save potentials for all elements to datavector
    void savePsi(); ///< save potentials for all elements to datavector
    void saveFnEta();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveFpKsi();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveN(); ///< save electron concentrations for all elements to datavector
    void saveP(); ///< save hole concentrations for all elements to datavector

    /// Add corrections to datavectors
    template <CalcType calctype>
    double addCorr(DataVector<double>& corr, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& vconst);

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

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

//     /// Save locate stiffness matrix to global one
//     inline void addCurvature(double& k44, double& k33, double& k22, double& k11,
//                              double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
//                              double ky, double width, const Vec<2,double>& midpoint);

    /// Set stiffness matrix + load vector
    template <CalcType calctype, typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(unsigned loops=1);

  public:

    double maxerr;              ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition      
    BoundaryConditions<RectangularMesh<2>,double> voltage_boundary;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<QuasiFermiEnergyLevelForElectrons, Geometry2DType>::Delegate outQuasiFermiEnergyLevelForElectrons;

    typename ProviderFor<QuasiFermiEnergyLevelForHoles, Geometry2DType>::Delegate outQuasiFermiEnergyLevelForHoles;

    typename ProviderFor<ConductionBandEdge, Geometry2DType>::Delegate outConductionBandEdge;

    typename ProviderFor<ValenceBandEdge, Geometry2DType>::Delegate outValenceBandEdge;

/*
    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;

    typename ProviderFor<Conductivity, Geometry2DType>::Delegate outConductivity;
*/
    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    Algorithm algorithm;    ///< Factorization algorithm to use

    double maxerrPsiI;  ///< Maximum estimated error for initial potential during all iterations (useful for single calculations managed by external python script)
    double maxerrPsi0;  ///< Maximum estimated error for potential at U = 0 V during all iterations (useful for single calculations managed by external python script)
    double maxerrPsi;   ///< Maximum estimated error for potential during all iterations (useful for single calculations managed by external python script)
    double maxerrFn;    ///< Maximum estimated error for quasi-Fermi energy level for electrons during all iterations (useful for single calculations managed by external python script)
    double maxerrFp;    ///< Maximum estimated error for quasi-Fermi energy level for holes during all iterations (useful for single calculations managed by external python script)
    size_t loopsPsiI;   ///< Loops limit for initial potential
    size_t loopsPsi0;   ///< Loops limit for potential at U = 0 V
    size_t loopsPsi;    ///< Loops limit for potential
    size_t loopsFn;     ///< Loops limit for quasi-Fermi energy level for electrons
    size_t loopsFp;     ///< Loops limit for quasi-Fermi energy level for holes
    double itererr;     ///< Allowed residual iteration for iterative method
    size_t iterlim;     ///< Maximum number of iterations for iterative method
    size_t logfreq;     ///< Frequency of iteration progress reporting

    /**
     * Run drift_diffusion calculations
     * \return max correction of potential against the last call
     */
    double compute(unsigned loops=1);

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

    const LazyData<double> getQuasiFermiEnergyLevelsForElectrons(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getQuasiFermiEnergyLevelsForHoles(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getConductionBandEdges(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<double> getValenceBandEdges(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);


    /*
    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensities(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);

    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2> > dest_mesh, InterpolationMethod method);
*/
};

}} //namespaces

} // namespace plask

#endif

