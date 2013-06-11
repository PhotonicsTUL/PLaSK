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

    int mAsize;            ///< Number of columns in the main matrix

    double mJs;             ///< p-n junction parameter [A/m^2]
    double mBeta;           ///< p-n junction parameter [1/V]
    double mCondPcontact;   ///< p-contact electrical conductivity [S/m]
    double mCondNcontact;   ///< n-contact electrical conductivity [S/m]

    double mVCorrLim;       ///< Maximum voltage correction accepted as convergence
    int mLoopNo;            ///< Number of completed loops
    double mMaxAbsVCorr;    ///< Maximum absolute voltage correction (useful for single calculations managed by external python script)
    double mMaxRelVCorr;    ///< Maximum relative voltage correction (useful for single calculations managed by external python script)
    double mMaxVCorr;       ///< Maximum absolute voltage correction (useful for calculations with internal loops)
    double mDV;             ///< Maximum voltage

    DataVector<double> mCondJunc;                   ///< electrical conductivity for p-n junction in y-direction [S/m]

    DataVector<Tensor2<double>> mCond;              ///< Cached element conductivities
    DataVector<double> mPotentials;                 ///< Computed potentials
    DataVector<Vec<2,double>> mCurrentDensities;    ///< Computed current densities
    DataVector<double> mHeatDensities;              ///< Computed and cached heat source densities

    std::vector<size_t>
        mActLo,                ///< Vertical index of the lower side of the active regions
        mActHi;                ///< Vertical index of the higher side of the active regions
    std::vector<double> mDact; ///< Active regions thickness

    /// Save locate stiffness matrix to global one
    inline void setLocalMatrix(double& ioK44, double& ioK33, double& ioK22, double& ioK11,
                               double& ioK43, double& ioK21, double& ioK42, double& ioK31, double& ioK32, double& ioK41,
                               double iKy, double iElemWidth, const Vec<2,double>& iMidPoint);

    /// Load conductivities
    void loadConductivities();

    /// Save conductivities of active region
    void saveConductivities();

    /// Update stored potentials and calculate corrections
    void savePotentials(DataVector<double>& iV);

    /// Update stored current densities
    void saveCurrentDensities();

    /// Create 2D-vector with calculated heat densities
    void saveHeatDensities();

    /// Matrix solver
    void solveMatrix(DpbMatrix& iA, DataVector<double>& ioB);

    /// Matrix solver
    void solveMatrix(DgbMatrix& iA, DataVector<double>& ioB);

    /// Matrix solver
    void solveMatrix(SparseBandMatrix& iA, DataVector<double>& ioB);

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
    void setMatrix(MatrixT& oA, DataVector<double>& oLoad, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst);

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(int iLoopLim=1);

  public:

    CorrectionType mCorrType; ///< Type of the returned correction

    HeatMethod mHeatMethod; ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectilinearMesh2D,double> mVConst;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<CurrentDensity, Geometry2DType>::Delegate outCurrentDensity;

    typename ProviderFor<HeatDensity, Geometry2DType>::Delegate outHeatDensity;

    ReceiverFor<Temperature, Geometry2DType> inTemperature;

    ReceiverFor<Wavelength> inWavelength; /// wavelength (for heat generation in the active region) [nm]

    Algorithm mAlgorithm;   ///< Factorization algorithm to use

    double mIterErr;        ///< Allowed residual iteration for iterative method
    size_t mIterLim;        ///< Maximum nunber of iterations for iterative method
    size_t mLogFreq;        ///< Frequency of iteration progress reporting

    /**
     * Run electrical calculations
     * \return max correction of potential against the last call
     **/
    double compute(int iLoopLim=1);

    /**
     * Integrate vertical total current at certain level.
     * \param iVertIndex vertical index of the element smesh to perform integration at
     * \return computed total current
     */
    double integrateCurrent(size_t iVertIndex);

    /**
     * Integrate vertical total current flowing vertically through active region
     * \param iNact number of the active region
     * \return computed total current
     */
    double getTotalCurrent(size_t iNact=0);

    /// \return max absolute correction for potential
    double getMaxAbsVCorr() const { return mMaxAbsVCorr; } // result in [K]

    /// \return get max relative correction for potential
    double getMaxRelVCorr() const { return mMaxRelVCorr; } // result in [%]

    double getVCorrLim() const { return mVCorrLim; }
    void setVCorrLim(double iVCorrLim) { mVCorrLim = iVCorrLim; }

    double getBeta() const { return mBeta; }
    void setBeta(double iBeta)  { mBeta = iBeta; }

    double getJs() const { return mJs; }
    void setJs(double iJs)  { mJs = iJs; }

    double getCondPcontact() const { return mCondPcontact; }
    void setCondPcontact(double iCondPcontact)  { mCondPcontact = iCondPcontact; }

    double getCondNcontact() const { return mCondNcontact; }
    void setCondNcontact(double iCondNcontact)  { mCondNcontact = iCondNcontact; }

    DataVector<const double> getCondJunc() const { return mCondJunc; }
    void setCondJunc(double iCond)  { mCondJunc.reset(mCondJunc.size(), iCond); }
    void setCondJunc(const DataVector<const double>& iCond)  {
        if (!this->mesh || iCond.size() != (this->mesh->axis0.size()-1) * getActNo())
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        mCondJunc = iCond.claim();
    }

    double getActLo(size_t iActN) const { return mActLo[iActN]; }
    double getActHi(size_t iActN) const { return mActHi[iActN]; }
    size_t getActNo() const { return mDact.size(); }

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

