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

    size_t mActLo,  ///< Vertical index of the lower side fo the active region
           mActHi;  ///< Vertical index of the higher side fo the active region
    double mDact;   ///< Active region thickness

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
    void setActiveRegion();

    virtual void onMeshChange(const typename RectilinearMesh2D::Event& evt) override {
        this->invalidate();
        setActiveRegion();
    }

    virtual void onGeometryChange(const Geometry::Event& evt) override {
        SolverWithMesh<Geometry2DType, RectilinearMesh2D>::onGeometryChange(evt);
        setActiveRegion();
    }


  public:

    CorrectionType mCorrType; ///< Type of the returned correction

    HeatMethod mHeatMethod; ///< Method of heat computation

    /// Boundary condition
    BoundaryConditions<RectilinearMesh2D,double> mVConst;

    typename ProviderFor<Potential, Geometry2DType>::Delegate outPotential;

    typename ProviderFor<CurrentDensity2D, Geometry2DType>::Delegate outCurrentDensity;

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
     * \return computed total current
     */
    double getTotalCurrent();

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
        if (!this->mesh || iCond.size() != this->mesh->axis0.size()-1)
            throw BadInput(this->getId(), "Provided junction conductivity vector has wrong size");
        mCondJunc = iCond.claim();
    }

    double getActLo() const { return mActLo; }
    double getActHi() const { return mActHi; }

    virtual void loadConfiguration(XMLReader& source, Manager& manager); // for solver configuration (see: *.xpl file with structures)

    FiniteElementMethodElectrical2DSolver(const std::string& name="");

    virtual std::string getClassName() const override;

    ~FiniteElementMethodElectrical2DSolver();

  protected:

    DataVector<const double> getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const;

    DataVector<const double> getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);

    DataVector<const Vec<2>> getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method);


    template <typename MatrixT>
    void applyBC(MatrixT& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
        // boundary conditions of the first kind
        for (auto cond: bvoltage) {
            for (auto r: cond.place) {
                A(r,r) = 1.;
                register double val = B[r] = cond.value;
                size_t start = (r > A.kd)? r-A.kd : 0;
                size_t end = (r + A.kd < A.size)? r+A.kd+1 : A.size;
                for(size_t c = start; c < r; ++c) {
                    B[c] -= A(r,c) * val;
                    A(r,c) = 0.;
                }
                for(size_t c = r+1; c < end; ++c) {
                    B[c] -= A(r,c) * val;
                    A(r,c) = 0.;
                }
            }
        }
    }

    void applyBC(SparseBandMatrix& A, DataVector<double>& B, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
        // boundary conditions of the first kind
        for (auto cond: bvoltage) {
            for (auto r: cond.place) {
                double* rdata = A.data + LDA*r;
                *rdata = 1.;
                register double val = B[r] = cond.value;
                // below diagonal
                for (register ptrdiff_t i = 4; i > 0; --i) {
                    register ptrdiff_t c = r - A.bno[i];
                    if (c >= 0) {
                        B[c] -= A.data[LDA*c+i] * val;
                        A.data[LDA*c+i] = 0.;
                    }
                }
                // above diagonal
                for (register ptrdiff_t i = 1; i < 5; ++i) {
                    register ptrdiff_t c = r + A.bno[i];
                    if (c < A.size) {
                        B[c] -= rdata[i] * val;
                        rdata[i] = 0.;
                    }
                }
            }
        }
    }

    /// Set stiffness matrix + load vector
    template <typename MatrixT>
    void setMatrix(MatrixT& oA, DataVector<double>& oLoad, const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst)
    {
        this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, bands=%2%{%3%})", oA.size, oA.kd+1, oA.ld+1);

        std::fill_n(oA.data, oA.size*(oA.ld+1), 0.); // zero the matrix
        oLoad.fill(0.);

        std::vector<Box2D> tVecBox = this->geometry->getLeafsBoundingBoxes();

        // Set stiffness matrix and load vector
        for (auto tE: this->mesh->elements)
        {
            size_t i = tE.getIndex();

            // nodes numbers for the current element
            size_t tLoLeftNo = tE.getLoLoIndex();
            size_t tLoRghtNo = tE.getUpLoIndex();
            size_t tUpLeftNo = tE.getLoUpIndex();
            size_t tUpRghtNo = tE.getUpUpIndex();

            // element size
            double tElemWidth = tE.getUpper0() - tE.getLower0();
            double tElemHeight = tE.getUpper1() - tE.getLower1();

            Vec<2,double> tMidPoint = tE.getMidpoint();

            // update junction conductivities
            if (mLoopNo != 0 && this->geometry->hasRoleAt("active", tMidPoint)) {
                size_t tLeft = this->mesh->index0(tLoLeftNo);
                size_t tRight = this->mesh->index0(tLoRghtNo);
                double tJy = 0.5e6 * mCond[i].c11 *
                    abs( - mPotentials[this->mesh->index(tLeft, mActLo)] - mPotentials[this->mesh->index(tRight, mActLo)]
                        + mPotentials[this->mesh->index(tLeft, mActHi)] + mPotentials[this->mesh->index(tRight, mActHi)]
                    ) / mDact; // [j] = A/m²
                mCond[i] = Tensor2<double>(0., 1e-6 * mBeta * tJy * mDact / log(tJy / mJs + 1.));
            }
            double tKx = mCond[i].c00;
            double tKy = mCond[i].c11;

            tKx *= tElemHeight; tKx /= tElemWidth;
            tKy *= tElemWidth; tKy /= tElemHeight;

            // set symmetric matrix components
            double tK44, tK33, tK22, tK11, tK43, tK21, tK42, tK31, tK32, tK41;

            tK44 = tK33 = tK22 = tK11 = (tKx + tKy) / 3.;
            tK43 = tK21 = (-2. * tKx + tKy) / 6.;
            tK42 = tK31 = - (tKx + tKy) / 6.;
            tK32 = tK41 = (tKx - 2. * tKy) / 6.;

            // set stiffness matrix
            setLocalMatrix(tK44, tK33, tK22, tK11, tK43, tK21, tK42, tK31, tK32, tK41, tKy, tElemWidth, tMidPoint);

            oA(tLoLeftNo, tLoLeftNo) += tK11;
            oA(tLoRghtNo, tLoRghtNo) += tK22;
            oA(tUpRghtNo, tUpRghtNo) += tK33;
            oA(tUpLeftNo, tUpLeftNo) += tK44;

            oA(tLoRghtNo, tLoLeftNo) += tK21;
            oA(tUpRghtNo, tLoLeftNo) += tK31;
            oA(tUpLeftNo, tLoLeftNo) += tK41;
            oA(tUpRghtNo, tLoRghtNo) += tK32;
            oA(tUpLeftNo, tLoRghtNo) += tK42;
            oA(tUpLeftNo, tUpRghtNo) += tK43;
        }

        // boundary conditions of the first kind
        applyBC(oA, oLoad, iVConst);

    #ifndef NDEBUG
        double* tAend = oA.data + oA.size * oA.kd;
        for (double* pa = oA.data; pa != tAend; ++pa) {
            if (isnan(*pa) || isinf(*pa))
                throw ComputationError(this->getId(), "Error in stiffness matrix at position %1% (%2%)", pa-oA.data, isnan(*pa)?"nan":"inf");
        }
    #endif

    }

    /// Perform computations for particular matrix type
    template <typename MatrixT>
    double doCompute(int iLoopLim=1)
    {
        this->initCalculation();

        mCurrentDensities.reset();
        mHeatDensities.reset();

        // Store boundary conditions for current mesh
        auto tVConst = mVConst(this->mesh);

        this->writelog(LOG_INFO, "Running electrical calculations");

        int tLoop = 0;
        MatrixT tA(mAsize, this->mesh->minorAxis().size());

        double tMaxMaxAbsVCorr = 0.,
            tMaxMaxRelVCorr = 0.;

    #   ifndef NDEBUG
            if (!mPotentials.unique()) this->writelog(LOG_DEBUG, "Potential data held by something else...");
    #   endif
        mPotentials = mPotentials.claim();
        DataVector<double> tV(mAsize);

        loadConductivities();

        do {
            setMatrix(tA, tV, tVConst);

            solveMatrix(tA, tV);

            savePotentials(tV);

            if (mMaxAbsVCorr > tMaxMaxAbsVCorr) tMaxMaxAbsVCorr = mMaxAbsVCorr;

            ++mLoopNo;
            ++tLoop;

            // show max correction
            this->writelog(LOG_RESULT, "Loop %d(%d): DeltaV=%.3fV, update=%.3fV(%.3f%%)", tLoop, mLoopNo, mDV, mMaxAbsVCorr, mMaxRelVCorr);

        } while (((mCorrType == CORRECTION_ABSOLUTE)? (mMaxAbsVCorr > mVCorrLim) : (mMaxRelVCorr > mVCorrLim)) && (iLoopLim == 0 || tLoop < iLoopLim));

        saveConductivities();

        outPotential.fireChanged();
        outCurrentDensity.fireChanged();
        outHeatDensity.fireChanged();

        // Make sure we store the maximum encountered values, not just the last ones
        // (so, this will indicate if the results changed since the last run, not since the last loop iteration)
        mMaxAbsVCorr = tMaxMaxAbsVCorr;
        mMaxRelVCorr = tMaxMaxRelVCorr;

        if (mCorrType == CORRECTION_RELATIVE) return mMaxRelVCorr;
        else return mMaxAbsVCorr;
    }


};

}} //namespaces

} // namespace plask

#endif

