#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    mJs(1.),
    mBeta(20.),
    mCondPcontact(5.),
    mCondNcontact(50.),
    mVCorrLim(1e-3),
    mLoopNo(0),
    mDefCondJunc(5.),
    mCorrType(CORRECTION_ABSOLUTE),
    mHeatMethod(HEAT_JOULES),
    outPotential(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    outHeatDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    mAlgorithm(ALGORITHM_CHOLESKY),
    mIterErr(1e-8),
    mIterLim(10000),
    mLogFreq(500)
{
    onInvalidate();
    inTemperature = 300.;
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "voltage" || param == "potential")
            this->readBoundaryConditions(manager, source, mVConst);

        else if (param == "loop") {
            mVCorrLim = source.getAttribute<double>("corrlim", mVCorrLim);
            mCorrType = source.enumAttribute<CorrectionType>("corrtype")
                .value("absolute", CORRECTION_ABSOLUTE, 3)
                .value("relative", CORRECTION_RELATIVE, 3)
                .get(mCorrType);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            mAlgorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(mAlgorithm);
            mIterErr = source.getAttribute<double>("itererr", mIterErr);
            mIterLim = source.getAttribute<size_t>("iterlim", mIterLim);
            mLogFreq = source.getAttribute<size_t>("logfreq", mLogFreq);
            source.requireTagEnd();
        }

        else if (param == "junction") {
            mJs = source.getAttribute<double>("js", mJs);
            mBeta = source.getAttribute<double>("beta", mBeta);
            auto tCondJunc = source.getAttribute<double>("pnjcond");
            if (tCondJunc) setCondJunc(*tCondJunc);
            auto tWavelength = source.getAttribute<double>("wavelength");
            if (tWavelength) inWavelength = *tWavelength;
            mHeatMethod = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(mHeatMethod);
            source.requireTagEnd();
        }

        else if (param == "contacts") {
            mCondPcontact = source.getAttribute<double>("pcond", mCondPcontact);
            mCondNcontact = source.getAttribute<double>("ncond", mCondNcontact);
            source.requireTagEnd();
        }

        else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::~FiniteElementMethodElectrical2DSolver() {
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setActiveRegions()
{
    if (!this->geometry || !this->mesh) {
        if (mCondJunc.size() != 1) {
            double tCondY = 0.;
            for (auto cond: mCondJunc) tCondY += cond;
            mCondJunc.reset(1, tCondY / mCondJunc.size());
        }
        return;
    }

    mActLo.clear();
    mActHi.clear();
    mDact.clear();

    shared_ptr<RectilinearMesh2D> points = this->mesh->getMidpointsMesh();

    size_t ileft = 0, iright = points->axis0.size();
    bool in_active = false;

    for (size_t r = 0; r < points->axis1.size(); ++r) {
        bool had_active = false; // indicates if we had active region in this layer
        shared_ptr<Material> layer_material;

        for (size_t c = 0; c < points->axis0.size(); ++c) { // In the (possible) active region
            auto point = points->at(c,r);
            bool active = (bool)this->geometry->hasRoleAt("active", point);

            if (c < ileft) {
                if (active)
                    throw Exception("%1%: Left edge of the active region not aligned.", this->getId());
            } else if (c >= iright) {
                if (active)
                    throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
            } else {
                // Here we are inside potential active region
                if (active) {
                    if (!had_active) {
                        if (!in_active) { // active region is starting set-up new region info
                            ileft = c;
                            mActLo.push_back(r);
                        }
                    }
                } else if (had_active) {
                    if (!in_active) {
                        iright = c;
                    } else
                        throw Exception("%1%: Right edge of the active region not aligned.", this->getId());
                }
                had_active |= active;
            }
        }
        in_active = had_active;

        // Test if the active region has finished
        if (!in_active && mActLo.size() != mActHi.size()) {
            mActHi.push_back(r);
            mDact.push_back(this->mesh->axis1[mActHi.back()] - this->mesh->axis1[mActLo.back()]);
            this->writelog(LOG_DETAIL, "Detected active layer %2% thickness = %1%nm", 1e3 * mDact.back(), mDact.size()-1);
        }
    }

    assert(mActHi.size() == mActLo.size());

    size_t tCondSize = max(mActLo.size() * (this->mesh->axis0.size()-1), size_t(1));

    if (mCondJunc.size() != tCondSize) {
        double tCondY = 0.;
        for (auto cond: mCondJunc) tCondY += cond;
        mCondJunc.reset(tCondSize, tCondY / mCondJunc.size());
    }
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    mLoopNo = 0;
    mAsize = this->mesh->size();
    mPotentials.reset(mAsize, 0.);
    mCond.reset(this->mesh->elements.size());
    if (mCondJunc.size() == 1) {
        size_t tCondSize = max(mActLo.size() * (this->mesh->axis0.size()-1), size_t(1));
        mCondJunc.reset(tCondSize, mCondJunc[0]);
    }
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInvalidate() {
    mCond.reset();
    mPotentials.reset();
    mCurrentDensities.reset();
    mHeatDensities.reset();
    mCondJunc.reset(1, mDefCondJunc);
}


template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::setLocalMatrix(double&, double&, double&, double&,
                            double&, double&, double&, double&, double&, double&,
                            double, double, const Vec<2,double>&) {
    return;
}

template<>
inline void FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::setLocalMatrix(double& ioK44, double& ioK33, double& ioK22, double& ioK11,
                               double& ioK43, double& ioK21, double& ioK42, double& ioK31, double& ioK32, double& ioK41,
                               double iKy, double iElemWidth, const Vec<2,double>& iMidPoint) {
        double r = iMidPoint.rad_r();
        double tKr = iKy * iElemWidth / 12.;
        ioK44 = r * ioK44 - tKr;
        ioK33 = r * ioK33 + tKr;
        ioK22 = r * ioK22 + tKr;
        ioK11 = r * ioK11 - tKr;
        ioK43 = r * ioK43;
        ioK21 = r * ioK21;
        ioK42 = r * ioK42;
        ioK31 = r * ioK31;
        ioK32 = r * ioK32 - tKr;
        ioK41 = r * ioK41 + tKr;
}

template<typename Geometry2DType> template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(MatrixT& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
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

template<typename Geometry2DType>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::applyBC(SparseBandMatrix& A, DataVector<double>& B,
                                                                    const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& bvoltage) {
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
template<typename Geometry2DType> template <typename MatrixT>
void FiniteElementMethodElectrical2DSolver<Geometry2DType>::setMatrix(MatrixT& oA, DataVector<double>& oLoad,
                                                                      const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst)
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
            size_t tNact = std::upper_bound(mActHi.begin(), mActHi.end(), this->mesh->index1(tLoLeftNo)) - mActHi.begin();
            assert(tNact < mActHi.size());
            double tJy = 0.5e6 * mCond[i].c11 *
                abs( - mPotentials[this->mesh->index(tLeft, mActLo[tNact])] - mPotentials[this->mesh->index(tRight, mActLo[tNact])]
                    + mPotentials[this->mesh->index(tLeft, mActHi[tNact])] + mPotentials[this->mesh->index(tRight, mActHi[tNact])]
                ) / mDact[tNact]; // [j] = A/m²
            mCond[i] = Tensor2<double>(0., 1e-6 * mBeta * tJy * mDact[tNact] / log(tJy / mJs + 1.));
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


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::loadConductivities()
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tTemperature = inTemperature(iMesh);

    for (auto tE: this->mesh->elements)
    {
        size_t i = tE.getIndex();
        Vec<2,double> tMidPoint = tE.getMidpoint();

        auto tRoles = this->geometry->getRolesAt(tMidPoint);
        if (tRoles.find("active") != tRoles.end()) {
            size_t n = std::upper_bound(mActHi.begin(), mActHi.end(), this->mesh->index1(i)) - mActHi.begin();
            assert(n < mActHi.size());
            mCond[i] = Tensor2<double>(0., mCondJunc[n * (this->mesh->axis0.size()-1) + tE.getIndex0()]);
        } else if (tRoles.find("p-contact") != tRoles.end()) {
            mCond[i] = Tensor2<double>(mCondPcontact, mCondPcontact);
        } else if (tRoles.find("n-contact") != tRoles.end()) {
            mCond[i] = Tensor2<double>(mCondNcontact, mCondNcontact);
        } else
            mCond[i] = this->geometry->getMaterial(tMidPoint)->cond(tTemperature[i]);
    }
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveConductivities()
{
    for (size_t n = 0; n < getActNo(); ++n)
        for (size_t i = 0, j = (mActLo[n]+mActHi[n])/2; i != this->mesh->axis0.size()-1; ++i)
            mCondJunc[n * (this->mesh->axis0.size()-1) + i] = mCond[this->mesh->elements(i,j).getIndex()].c11;
}


template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::compute(int loops) {
    switch (mAlgorithm) {
        case ALGORITHM_CHOLESKY: return doCompute<DpbMatrix>(loops);
        case ALGORITHM_GAUSS: return doCompute<DgbMatrix>(loops);
        case ALGORITHM_ITERATIVE: return doCompute<SparseBandMatrix>(loops);
    }
    return 0.;
}

template<typename Geometry2DType> template <typename MatrixT>
double FiniteElementMethodElectrical2DSolver<Geometry2DType>::doCompute(int iLoopLim)
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
        if (mMaxRelVCorr > tMaxMaxRelVCorr) tMaxMaxRelVCorr = mMaxRelVCorr;

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

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DpbMatrix& iA, DataVector<double>& ioB)
{
    int info = 0;

    this->writelog(LOG_DETAIL, "Solving matrix system");

    // Factorize matrix
    dpbtrf(UPLO, iA.size, iA.kd, iA.data, iA.ld+1, info);
    if (info < 0)
        throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
    else if (info > 0)
        throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);

    // Find solutions
    dpbtrs(UPLO, iA.size, iA.kd, 1, iA.data, iA.ld+1, ioB.data(), ioB.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(DgbMatrix& iA, DataVector<double>& ioB)
{
    int info = 0;
    this->writelog(LOG_DETAIL, "Solving matrix system");
    int* ipiv = aligned_malloc<int>(iA.size);

    iA.mirror();

    // Factorize matrix
    dgbtrf(iA.size, iA.size, iA.kd, iA.kd, iA.data, iA.ld+1, ipiv, info);
    if (info < 0) {
        aligned_free(ipiv);
        throw CriticalException("%1%: Argument %2% of dgbtrf has illegal value", this->getId(), -info);
    } else if (info > 0) {
        aligned_free(ipiv);
        throw ComputationError(this->getId(), "Matrix is singlar (at %1%)", info);
    }

    // Find solutions
    dgbtrs('N', iA.size, iA.kd, iA.kd, 1, iA.data, iA.ld+1, ipiv, ioB.data(), ioB.size(), info);
    aligned_free(ipiv);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dgbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(SparseBandMatrix& ioA, DataVector<double>& ioB)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    PrecondJacobi precond(ioA);

    DataVector<double> tX = mPotentials.copy(); // We use previous potentials as initial solution
    double tErr;
    try {
        int iter = solveDCG(ioA, precond, tX.data(), ioB.data(), tErr, mIterLim, mIterErr, mLogFreq, this->getId());
        this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
    } catch (DCGError tExc) {
        throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", tExc.what());
    }

    ioB = tX;

    // now A contains factorized matrix and B the solutions
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::savePotentials(DataVector<double>& iV)
{
    mMaxAbsVCorr = 0.;
    mMaxRelVCorr = 0.;

    double tMaxV = 0., tMinV = INFINITY;

    for (auto ttPot = mPotentials.begin(), ttV = iV.begin(); ttV != iV.end(); ++ttPot, ++ttV)
    {
        double tAbsCorr = std::abs(*ttV - *ttPot); // for boundary with constant temperature this will be zero anyway
        double tRelCorr = tAbsCorr / *ttV;
        if (tAbsCorr > mMaxAbsVCorr) mMaxAbsVCorr = tAbsCorr;
        if (tRelCorr > mMaxRelVCorr) mMaxRelVCorr = tRelCorr;
        if (*ttV > tMaxV) tMaxV = *ttV;
        if (*ttV < tMinV) tMinV = *ttV;
    }
    mDV = tMaxV - tMinV;
    mMaxRelVCorr *= 100.; // %
    if (mLoopNo == 0) mMaxRelVCorr = 100.;
    std::swap(mPotentials, iV);
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveCurrentDensities()
{
    this->writelog(LOG_DETAIL, "Computing current densities");

    mCurrentDensities.reset(this->mesh->elements.size());

    for (auto tE: this->mesh->elements) {
        size_t i = tE.getIndex();
        size_t tLoLeftNo = tE.getLoLoIndex();
        size_t tLoRghtNo = tE.getUpLoIndex();
        size_t tUpLeftNo = tE.getLoUpIndex();
        size_t tUpRghtNo = tE.getUpUpIndex();
        double tDVx = - 0.05 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                             / (tE.getUpper0() - tE.getLower0()); // [j] = kA/cm²
        double tDVy = - 0.05 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                             / (tE.getUpper1() - tE.getLower1()); // [j] = kA/cm²
        mCurrentDensities[i] = vec(mCond[i].c00 * tDVx, mCond[i].c11 * tDVy);
    }
}

template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveHeatDensities()
{
    this->writelog(LOG_DETAIL, "Computing heat densities");

    mHeatDensities.reset(this->mesh->elements.size());

    if (mHeatMethod == HEAT_JOULES) {
        for (auto tE: this->mesh->elements) {
            size_t i = tE.getIndex();
            size_t tLoLeftNo = tE.getLoLoIndex();
            size_t tLoRghtNo = tE.getUpLoIndex();
            size_t tUpLeftNo = tE.getLoUpIndex();
            size_t tUpRghtNo = tE.getUpUpIndex();
            auto tMidPoint = tE.getMidpoint();
            if (this->geometry->getMaterial(tMidPoint)->kind() == Material::NONE || this->geometry->hasRoleAt("noheat", tMidPoint))
                mHeatDensities[i] = 0.;
            else {
                double tDVx = 0.5e6 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                    / (tE.getUpper0() - tE.getLower0()); // [grad(dV)] = V/m
                double tDVy = 0.5e6 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                    / (tE.getUpper1() - tE.getLower1()); // [grad(dV)] = V/m
                mHeatDensities[i] = mCond[i].c00 * tDVx*tDVx + mCond[i].c11 * tDVy*tDVy;
            }
        }
    } else {
        for (auto tE: this->mesh->elements) {
            size_t i = tE.getIndex();
            size_t tLoLeftNo = tE.getLoLoIndex();
            size_t tLoRghtNo = tE.getUpLoIndex();
            size_t tUpLeftNo = tE.getLoUpIndex();
            size_t tUpRghtNo = tE.getUpUpIndex();
            double tDVx = 0.5e6 * (- mPotentials[tLoLeftNo] + mPotentials[tLoRghtNo] - mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                / (tE.getUpper0() - tE.getLower0()); // [grad(dV)] = V/m
            double tDVy = 0.5e6 * (- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                / (tE.getUpper1() - tE.getLower1()); // [grad(dV)] = V/m
            auto tMidPoint = tE.getMidpoint();
            auto tRoles = this->geometry->getRolesAt(tMidPoint);
            if (tRoles.find("active") != tRoles.end()) {
                size_t tNact = std::upper_bound(mActHi.begin(), mActHi.end(), this->mesh->index1(i)) - mActHi.begin();
                assert(tNact < mActHi.size());
                double tHeatFact = 1e15 * phys::h_J * phys::c / (phys::qe * real(inWavelength(0)) * mDact[tNact]);
                double tJy = mCond[i].c11 * fabs(tDVy); // [j] = A/m²
                mHeatDensities[i] = tHeatFact * tJy ;
            } else if (this->geometry->getMaterial(tMidPoint)->kind() == Material::NONE || tRoles.find("noheat") != tRoles.end())
                mHeatDensities[i] = 0.;
            else
                mHeatDensities[i] = mCond[i].c00 * tDVx*tDVx + mCond[i].c11 * tDVy*tDVy;
        }
    }
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::integrateCurrent(size_t iVertIndex)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    if (!mCurrentDensities) saveCurrentDensities();
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, iVertIndex);
        result += mCurrentDensities[element.getIndex()].c1 * element.getSize0();
    }
    result *= geometry->getExtrusion()->getLength() * 0.01; // kA/cm² µm² -->  mA
    if (this->getGeometry()->isSymmetric(Geometry::DIRECTION_TRAN)) return 2.*result;
    return result;
}


template<> double FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::integrateCurrent(size_t iVertIndex)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Computing total current");
    if (!mCurrentDensities) saveCurrentDensities();
    double result = 0.;
    for (size_t i = 0; i < mesh->axis0.size()-1; ++i) {
        auto element = mesh->elements(i, iVertIndex);
        double rin = element.getLower0(), rout = element.getUpper0();
        result += mCurrentDensities[element.getIndex()].c1 * (rout*rout - rin*rin);
    }
    return result * M_PI * 0.01; // kA/cm² µm² -->  mA
}


template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::getTotalCurrent(size_t iNact)
{
    if (iNact >= mActLo.size()) throw BadInput(this->getId(), "Wrong active region number");
    // Find the average of the active region
    size_t level = (mActLo[iNact] + mActHi[iNact]) / 2;
    return integrateCurrent(level);
}


template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials(const MeshD<2>& dst_mesh, InterpolationMethod method) const
{
    if (!mPotentials) throw NoValue("Potentials");
    this->writelog(LOG_DETAIL, "Getting potentials");
    if (method == DEFAULT_INTERPOLATION)  method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mPotentials, WrappedMesh<2>(dst_mesh, this->geometry), method);
}


template<typename Geometry2DType> DataVector<const Vec<2> > FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!mPotentials) throw NoValue("Current densities");
    this->writelog(LOG_DETAIL, "Getting current densities");
    if (!mCurrentDensities) saveCurrentDensities();
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), mCurrentDensities, dest_mesh, method);
    constexpr Vec<2> zero(0.,0.);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = zero;
    return result;
}

template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!mPotentials) throw NoValue("Heat densities");
    this->writelog(LOG_DETAIL, "Getting heat densities");
    if (!mHeatDensities) saveHeatDensities(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    auto dest_mesh = WrappedMesh<2>(dst_mesh, this->geometry);
    auto result = interpolate(*(this->mesh->getMidpointsMesh()), mHeatDensities, dest_mesh, method);
    for (size_t i = 0; i < result.size(); ++i)
        if (!this->geometry->getChildBoundingBox().contains(dest_mesh[i])) result[i] = 0.;
    return result;
}



template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Beta2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.BetaCyl"; }

template struct FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
