#ifndef PLASK__MODULE_ELECTRICAL_DDM2D_H
#define PLASK__MODULE_ELECTRICAL_DDM2D_H

#include <plask/plask.hpp>
#include <limits>

PLASK_NO_CONVERSION_WARNING_BEGIN
#include <Eigen/Eigen>
PLASK_NO_WARNING_END
#include "fd.h"
#include "block_matrix.h"
#include "iterative_matrix.h"
#include "gauss_matrix.h"
#include <fstream>

namespace plask { namespace electrical { namespace drift_diffusion {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

/// Carrier statistics types
enum Stat {
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

/// Contact types
enum ContType {
    OHMIC,              ///< Ohmic contacts
    SCHOTTKY            ///< Schottky contacts
};

/**
 * Solver performing calculations in 2D Cartesian or Cylindrical space using finite element method
 */
template<typename Geometry2DType>
struct PLASK_SOLVER_API DriftDiffusionModel2DSolver: public SolverWithMesh<Geometry2DType, RectangularMesh<2>> {

  protected:

    std::size_t size;                   ///< Number of columns in the main matrix

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
    //double mtx;    ///< SRH recombination lifetime (s)
    double mAx;    ///< radiative recombination coefficient (1/s)
    double mBx;    ///< radiative recombination coefficient (cm^3/s)
    double mCx;    ///< Auger recombination coefficient (cm^6/s)
    //double mHx;    ///< heat source (W/(m^3))
    double mPx;    ///< polarization (C/m^2)

    double dU;         ///< default voltage step (V)
    double maxDelPsi0; ///< maximal correction for initial potential calculations (V)
    double maxDelPsi;  ///< maximal correction for potential calculations (V)
    double maxDelFn;   ///< maximal correction for quasi-Fermi levels for electrons calculations (eV)
    double maxDelFp;   ///< maximal correction for quasi-Fermi levels for holes calculations (eV)

    Stat stat;  ///< carriers statistics
    ContType conttype; ///< type of contacts (ohmic/Schottky)

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
    DataVector<Vec<2,double>> currentsN;        ///< Computed current densities for electrons
    DataVector<Vec<2,double>> currentsP;        ///< Computed current densities for holes
    DataVector<double> heats;                   ///< Computed and cached heat source densities

    bool needPsi0;                             ///< Flag indicating if we need to compute initial potential;

    bool strained;                  ///< Consider strain in QW?

    /// Initialize the solver
    virtual void onInitialize() override;

    /// Invalidate the data
    virtual void onInvalidate() override;

    /// Get info on active region
    size_t getActiveRegionMeshIndex(size_t actnum) const;

    virtual void onMeshChange(const typename RectangularMesh<2>::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onMeshChange(evt);
    }

    virtual void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectangularMesh<2>>::onGeometryChange(evt);
    }

    /**
     * Calculate initial potential for all elements
     */
    void computePsiI();

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

  private:

    /// Slot called when gain has changed
    void onInputChange(ReceiverBase&, ReceiverBase::ChangeReason) {
        needPsi0 = true;
    }

    /// Find initial potential
    double findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT, std::size_t& loop) const;

    /**
    * Calculate electron concentration.
    * \param iNc effective density of states for the conduction band
    * \param iFnEta exponent of normalised quasi-Fermi energy level for elestrons
    * \param iPsi normalised energy (potential multiplied by the elementary charge)
    * \param iEc0 normalised conduction band edge
    * \param iT normalised temperature
    * \return computed electron concentration
    */
    double calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT) const {
        switch (stat) {
            //case STAT_MB: return ( iNc * iFnEta * exp(iPsi-iEc0) );
            case STAT_MB:
                //this->writelog(LOG_INFO, "Maxwell-Boltzmann statistics");
                return ( iNc * pow(iFnEta,1./iT) * exp((iPsi-iEc0)/iT) );
            //case STAT_FD: return ( iNc * fermiDiracHalf(log(iFnEta) + iPsi - iEc0) );
            case STAT_FD:
                //this->writelog(LOG_INFO, "Fermi-Dirac statistics");
                return ( iNc * fermiDiracHalf((log(iFnEta) + iPsi - iEc0)/iT) );
        }
        return NAN;
    }

    /**
    * Calculate hole concentration.
    * \param iNv effective density of states for the valence band
    * \param iFpKsi exponent of normalised quasi-Fermi energy level for holes
    * \param iPsi normalised energy (potential multiplied by the elementary charge)
    * \param iEv0 normalised valence band edge
    * \param iT normalised temperature
    * \return computed hole concentration
    */
    double calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT) const {
        switch (stat) {
            //case STAT_MB: return ( iNv * iFpKsi * exp(iEv0-iPsi) );
            case STAT_MB: return ( iNv * pow(iFpKsi,1./iT) * exp((iEv0-iPsi)/iT) );
            //case STAT_FD: return ( iNv * fermiDiracHalf(log(iFpKsi) - iPsi + iEv0) );
            case STAT_FD: return ( iNv * fermiDiracHalf((log(iFpKsi) - iPsi + iEv0)/iT) );
        }
        return NAN;
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

    void savePsi0(); ///< save potentials for all elements to datavector
    void savePsi(); ///< save potentials for all elements to datavector
    void saveFnEta();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveFpKsi();  ///< save exponent of quasi-Fermi electron level for all elements to datavector
    void saveN(); ///< save electron concentrations for all elements to datavector
    void saveP(); ///< save hole concentrations for all elements to datavector

    /// Add corrections to datavectors
    template <CalcType calctype>
    double addCorr(DataVector<double>& corr, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& vconst);


    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();

    /// Matrix solver
    void solveMatrix(DpbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(DgbMatrix& A, DataVector<double>& B);

    /// Matrix solver
    void solveMatrix(SparseBandMatrix& A, DataVector<double>& B);

    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    /// Save locate stiffness matrix to global one
    inline void addCurvature(double& k44, double& k33, double& k22, double& k11,
                              double& k43, double& k21, double& k42, double& k31, double& k32, double& k41,
                              double ky, double width, const Vec<2,double>& midpoint);

    /// Set stiffness matrix + load vector
    template <CalcType calctype, typename MatrixT>
    void setMatrix(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectangularMesh<2>,double>& bvoltage);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(unsigned loops=1);

    /// Set some parameters
    int setSomeParams();

    /// Set mesh for region where energy levels will be found
    int setMeshActive(double _z1, double _z2, double _dz);

    /// Set vector with CB for active region
    int setCBel(std::vector<double> _CBel);

    /// Set vector with Me for active region
    int setCBelM(std::vector<double> _CBelM);

    /// Find energy levels for electrons in CB
    int findCBelLev();

    /// Find band edges for claddings
    int findCladBE();

    /// Structure containing information about each active region
    struct ActiveRegionInfo {
        shared_ptr<StackContainer<2>> layers;   ///< Stack containing all layers in the active region
        Vec<2> origin;                          ///< Location of the active region stack origin

        ActiveRegionInfo(Vec<2> origin): layers(plask::make_shared<StackContainer<2>>()), origin(origin) {}

        /// Return number of layers in the active region with surrounding barriers
        size_t size() const
        {
            return layers->getChildrenCount();
        }

        /// Return material of \p n-th layer
        shared_ptr<Material> getLayerMaterial(size_t n) const
        {
            auto block = static_cast<Block<2>*>(static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild().get());
            if (auto m = block->singleMaterial()) return m;
            throw plask::Exception("FreeCarrierGainSolver requires solid layers.");
        }

        /// Return translated bounding box of \p n-th layer
        Box2D getLayerBox(size_t n) const
        {
            return static_cast<GeometryObjectD<2>*>(layers->getChildNo(n).get())->getBoundingBox() + origin;
        }

        /// Return \p true if given layer is quantum well
        bool isQW(size_t n) const
        {
            return static_cast<Translation<2>*>(layers->getChildNo(n).get())->getChild()->hasRole("QW");
        }

        /// Return bounding box of the whole active region
        Box2D getBoundingBox() const
        {
            return layers->getBoundingBox() + origin;
        }

        /// Return \p true if the point is in the active region
        bool contains(const Vec<2>& point) const {
            return getBoundingBox().contains(point);
        }

        /// Return \p true if given point is inside quantum well
        bool inQW(const Vec<2>& point) const {
            if (!contains(point)) return false;
            assert(layers->getChildForHeight(point.c1-origin.c1));
            return layers->getChildForHeight(point.c1-origin.c1)->getChild()->hasRole("QW");
        }

        double averageNr(double lam, double T, double conc=0.) const {
            double nr = 0.;
            for (size_t i = 0; i != materials.size(); ++i)
                if (isQW(i)) nr += thicknesses[i] * materials[i]->Nr(lam, T, conc).real();
            return nr / totalqw;
        }

        std::vector<shared_ptr<Material>> materials;///< All materials in the active region
        std::vector<double> thicknesses;            ///< Thicknesses of the layers in the active region
        std::vector<size_t> wells;                  ///< Division of the active region into separate quantum wells

        double total;                               ///< Total active region thickness [µm]
        double totalqw;                             ///< Total accepted quantum wells thickness [µm]
        double bottom;                              ///< Bottom spacer thickness [µm]
        double top;                                 ///< Top spacer thickness [µm]

        enum ConsideredHoles: unsigned {
            NO_HOLES = 0,
            HEAVY_HOLES = 1,
            LIGHT_HOLES = 2,
            BOTH_HOLES = 3
        } holes;                                    ///< Type of holes existing in the active region


        /**
         * Summarize active region, check for appropriateness and compute some values
         * \param solver solver
         */
        void summarize(const DriftDiffusionModel2DSolver<Geometry2DType>* solver);
    };

    /// Substrate material
    shared_ptr<Material> materialSubstrate;

    /// List of active regions
    std::vector<ActiveRegionInfo> regions;

    /// Detect active regions.
    void detectActiveRegions();

    /// Parameters used to calculate energy levels of electrons
    std::vector<double> CBel; /// vector with energy band diagram for electrons from CB (nm)
    std::vector<double> CBelM; /// vector with masses of electrons from CB (m0)
    std::vector<double> CBelLev; /// energy levels of electrons from CB
    std::vector<double> z; /// vector with mesh-node positions along z-axis (nm)
    double dz; /// step - distance between mesh-nodes (nm)
    int nz; /// z-mesh size (number of all nodes)
    int ne; /// z-mesh size (number of all elements)
    double hhm; /// hb*hb/m, unit: eV*nm*nm
    double hh2m; /// hb*hb/(2m), unit: eV*nm*nm
    std::vector<double> lev_el; /// energy levels of electrons
    int n_lev_el; /// number of energy levels of electrons
    double Ec_min, Ec_max; /// min./max. conduction bands edges for claddings (the same for both claddings); unit: eV
    double T; /// temperature; unit: K
    double Eupshift; /// bands have to be up-shifted - we want only positive values of energy levels; unit: eV

  public:

    double maxerr;              ///< Maximum relative current density correction accepted as convergence

    /// Boundary condition
    BoundaryConditions<RectangularMesh<2>,double> voltage_boundary;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<FermiLevels, Geometry2DType>::Delegate outFermiLevels;

    typename ProviderFor<BandEdges, Geometry2DType>::Delegate outBandEdges;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensityForElectrons;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensityForHoles;

    typename ProviderFor<CarriersConcentration, Geometry2DType>::Delegate outCarriersConcentration;

    typename ProviderFor<Heat, Geometry2DType>::Delegate outHeat;
/*
    typename ProviderFor<Conductivity, Geometry2DType>::Delegate outConductivity;
*/
    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    Algorithm algorithm;    ///< Factorization algorithm to use

    bool mRsrh;    ///< SRH recombination is taken into account
    bool mRrad;    ///< radiative recombination is taken into account
    bool mRaug;    ///< Auger recombination is taken into account
    bool mPol;     ///< polarization (GaN is the substrate)
    bool mFullIon; ///< dopant ionization = 100%

    double mSchottkyP;  ///< Schottky barrier for p-type contact (eV)
    double mSchottkyN;  ///< Schottky barrier for n-type contact (eV)

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
     * Find energy levels - TEST
     * \return nothing important
     */
    double findEnergyLevels();

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
    const LazyData<double> getPotentials(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getFermiLevels(FermiLevels::EnumType what, shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method) const;

    const LazyData<double> getBandEdges(BandEdges::EnumType what, shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<double> getHeatDensities(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensitiesForElectrons(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<Vec<2>> getCurrentDensitiesForHoles(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

    const LazyData<double> getCarriersConcentration(CarriersConcentration::EnumType what, shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);

/*
    const LazyData<Tensor2<double>> getConductivity(shared_ptr<const MeshD<2>> dest_mesh, InterpolationMethod method);
*/
};

}} //namespaces

} // namespace plask

#endif

