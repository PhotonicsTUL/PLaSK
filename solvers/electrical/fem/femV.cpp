#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2DType> FiniteElementMethodElectrical2DSolver<Geometry2DType>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    mBigNum(1e15),
    mJs(1.),
    mBeta(20.),
    mCondJuncX0(1e-6),
    mCondJuncY0(5.),
    mCondPcontact(5.),
    mCondNcontact(50.),
    mVCorrLim(1e-3),
    mLoopNo(0),
    mCorrType(CORRECTION_ABSOLUTE),
    mHeatMethod(HEAT_JOULES),
    outPotential(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    outHeatDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    mAlgorithm(ALGORITHM_BLOCK)
{
    mCond.reset();
    mPotentials.reset();
    mCurrentDensities.reset();
    mHeatDensities.reset();

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
            auto tCorrType = source.getAttribute("corrtype");
            if (tCorrType) {
                std::string tValue = *tCorrType; boost::algorithm::to_lower(tValue);
                if (tValue == "absolute" || tValue == "abs") mCorrType = CORRECTION_ABSOLUTE;
                else if (tValue == "relative" || tValue == "rel") mCorrType = CORRECTION_RELATIVE;
                else throw XMLBadAttrException(source, "corrtype", *tCorrType, + "\"abs[olute]\" or \"rel[ative]\"");
            }
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            mBigNum = source.getAttribute<double>("bignum", mBigNum);
            auto tAlgo = source.getAttribute("algorithm");
            if (tAlgo) {
                std::string tValue = *tAlgo; boost::algorithm::to_lower(tValue);
                if (tValue == "slow") mAlgorithm = ALGORITHM_SLOW;
                else if (tValue == "block") mAlgorithm = ALGORITHM_SLOW;
                //else if (tValue == "iterative") mAlgorithm = ALGORITHM_ITERATIVE;
                else throw XMLBadAttrException(source, "algorithm", *tAlgo, + "\"block\" or \"slow\"");
            }
            source.requireTagEnd();
        }

        else if (param == "junction") {
            mJs = source.getAttribute<double>("js", mJs);
            mBeta = source.getAttribute<double>("beta", mBeta);
            auto tCondJunc0 = source.getAttribute("pnjcond");
            if (tCondJunc0) {
                try {
                    auto tConds = splitString2(*tCondJunc0, ',');
                    boost::trim(tConds.first);
                    boost::trim(tConds.second);
                    if (tConds.second != "") {
                        mCondJuncX0 = boost::lexical_cast<double>(tConds.first);
                        mCondJuncY0 = boost::lexical_cast<double>(tConds.second);
                    } else
                        mCondJuncX0 = mCondJuncY0 = boost::lexical_cast<double>(tConds.first);
                } catch (boost::bad_lexical_cast) {
                    throw XMLBadAttrException(source, "pnjcond", *tCondJunc0);
                }
            }
            auto tWavelength = source.getAttribute<double>("wavelength");
            if (tWavelength) inWavelength = *tWavelength;
            auto tHeamMethod = source.getAttribute("heat");
            if (tHeamMethod) {
                std::string tValue = *tHeamMethod; boost::algorithm::to_lower(tValue);
                if (tValue == "joules") mHeatMethod = HEAT_JOULES;
                else if (tValue == "wavelength") mHeatMethod = HEAT_BANDGAP;
                else throw XMLBadAttrException(source, "heat", *tHeamMethod, + "\"joules\" or \"wavelength\"");
            }
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


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    mLoopNo = 0;
    mAOrder = this->mesh->size();
    mABand = this->mesh->minorAxis().size() + 2;
    mPotentials.reset(mAOrder, 0.);
    mCond.reset(this->mesh->elements.size());
}


template<typename Geometry2DType> void FiniteElementMethodElectrical2DSolver<Geometry2DType>::onInvalidate() {
    mCond.reset();
    mPotentials.reset();
    mCurrentDensities.reset();
    mHeatDensities.reset();
}


template<>
void FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::setMatrix(BandSymMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, band=%2%)", mAOrder, mABand);

    std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
    oLoad.fill(0.);

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

    // Set stiffness matrix and load vector
    for (auto tE: mesh->elements)
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
        if (mLoopNo != 0 && geometry->hasRoleAt("active", tMidPoint)) {
            auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
            double tDact = 1e-6 * tLeaf->getBoundingBox().height();
            double tJy = 0.5e6 * mCond[i].c11 * fabs(- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                                    / (tE.getUpper1() - tE.getLower1()); // [j] = A/m²
            mCond[i] = std::make_pair(mCondJuncX0, mBeta * tJy * tDact / log(tJy / mJs + 1.));
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
    for (auto tCond: iVConst) {
        for (auto tIndex: tCond.place) {
            oA(tIndex, tIndex) += mBigNum;
            oLoad[tIndex] += tCond.value * mBigNum;
        }
    }

#ifndef NDEBUG
    double* tAend = oA.data + oA.size * oA.band1;
    for (double* pa = oA.data; pa != tAend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position %1% (%2%)", pa-oA.data, isnan(*pa)?"nan":"inf");
    }
#endif

}


template<>
void FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::setMatrix(BandSymMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iVConst
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, band=%2%)", mAOrder, mABand);

    std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
    oLoad.fill(0.);

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

    // Set stiffness matrix and load vector
    for (auto tE: mesh->elements)
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
        double r = tMidPoint.rad_r();

        if (mLoopNo != 0 && geometry->hasRoleAt("active", tMidPoint)) {
            auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
            double tDact = 1e-6 * tLeaf->getBoundingBox().height();
            double tJy = 0.5e6 * mCond[i].c11 * fabs(- mPotentials[tLoLeftNo] - mPotentials[tLoRghtNo] + mPotentials[tUpLeftNo] + mPotentials[tUpRghtNo])
                                                    / (tE.getUpper1() - tE.getLower1()); // [j] = A/m²
            mCond[i] = std::make_pair(mCondJuncX0, mBeta * tJy * tDact / log(tJy / mJs + 1.));
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

        double tKr = tKy * tElemWidth / 12.;

        // set stiffness matrix
        oA(tLoLeftNo, tLoLeftNo) += r * tK11 - tKr;
        oA(tLoRghtNo, tLoRghtNo) += r * tK22 + tKr;
        oA(tUpRghtNo, tUpRghtNo) += r * tK33 + tKr;
        oA(tUpLeftNo, tUpLeftNo) += r * tK44 - tKr;

        oA(tLoRghtNo, tLoLeftNo) += r * tK21;
        oA(tUpRghtNo, tLoLeftNo) += r * tK31;
        oA(tUpLeftNo, tLoLeftNo) += r * tK41 + tKr;
        oA(tUpRghtNo, tLoRghtNo) += r * tK32 - tKr;
        oA(tUpLeftNo, tLoRghtNo) += r * tK42;
        oA(tUpLeftNo, tUpRghtNo) += r * tK43;
    }

    // boundary conditions of the first kind
    for (auto tCond: iVConst) {
        for (auto tIndex: tCond.place) {
            oA(tIndex, tIndex) += mBigNum;
            oLoad[tIndex] += tCond.value * mBigNum;
        }
    }

#ifndef NDEBUG
    double* tAend = oA.data + oA.size * oA.band1;
    for (double* pa = oA.data; pa != tAend; ++pa) {
        if (isnan(*pa) || isinf(*pa))
            throw ComputationError(this->getId(), "Error in stiffness matrix at position %1%", pa-oA.data);
    }
#endif

}

template<typename Geometry2DType>  void FiniteElementMethodElectrical2DSolver<Geometry2DType>::saveConductivities()
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tTemperature = inTemperature(iMesh);

    for (auto tE: this->mesh->elements)
    {
        size_t i = tE.getIndex();
        Vec<2,double> tMidPoint = tE.getMidpoint();

        auto tRoles = this->geometry->getRolesAt(tMidPoint);
        if (tRoles.find("active") != tRoles.end()) {
            if (mLoopNo == 0) mCond[i] = std::make_pair(mCondJuncX0, mCondJuncY0);
            // in other loops leave it as it was
        } else if (tRoles.find("p-contact") != tRoles.end()) {
            mCond[i] = std::make_pair(mCondPcontact, mCondPcontact);
        } else if (tRoles.find("n-contact") != tRoles.end()) {
            mCond[i] = std::make_pair(mCondNcontact, mCondNcontact);
        } else
            mCond[i] = this->geometry->getMaterial(tMidPoint)->cond(tTemperature[i]);
    }
}



template<typename Geometry2DType> double FiniteElementMethodElectrical2DSolver<Geometry2DType>::compute(int iLoopLim)
{
    this->initCalculation();

    mCurrentDensities.reset();
    mHeatDensities.reset();

    // store boundary conditions for current mesh
    auto tVConst = mVConst(this->mesh);

    this->writelog(LOG_INFO, "Running electrical calculations");

    int tLoop = 0;
    BandSymMatrix tA(mAOrder, mABand);

    double tMaxMaxAbsVCorr = 0.,
           tMaxMaxRelVCorr = 0.;

#   ifndef NDEBUG
        if (!mPotentials.unique()) this->writelog(LOG_DEBUG, "Potential data held by something else...");
#   endif
    mPotentials = mPotentials.claim();
    DataVector<double> tV(mAOrder);

    saveConductivities();

    do {
        setMatrix(tA, tV, tVConst);

        int tInfo = solveMatrix(tA, tV);

        if (tInfo > 0)
            throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", tInfo);

        savePotentials(tV);

        if (mMaxAbsVCorr > tMaxMaxAbsVCorr) tMaxMaxAbsVCorr = mMaxAbsVCorr;

        ++mLoopNo;
        ++tLoop;

        // show max correction
        this->writelog(LOG_RESULT, "Loop %d(%d): DeltaV=%.3fV, update=%.3fV(%.3f%%)", tLoop, mLoopNo, mDV, mMaxAbsVCorr, mMaxRelVCorr);

    } while (((mCorrType == CORRECTION_ABSOLUTE)? (mMaxAbsVCorr > mVCorrLim) : (mMaxRelVCorr > mVCorrLim)) && (iLoopLim == 0 || tLoop < iLoopLim));

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


template<typename Geometry2DType> int FiniteElementMethodElectrical2DSolver<Geometry2DType>::solveMatrix(BandSymMatrix& iA, DataVector<double>& ioB)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    switch (mAlgorithm) {
        case ALGORITHM_SLOW:
            dpbtf2(UPLO, iA.size, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtf2 has illegal value", this->getId(), -info);
            break;
        case ALGORITHM_BLOCK:
            dpbtrf(UPLO, iA.size, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
            break;
    }
    if (info > 0) return info;

    // Find solutions
    dpbtrs(UPLO, iA.size, iA.band1, 1, iA.data, iA.band1+1, ioB.data(), ioB.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
    return info;
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
    if (mLoopNo == 0) mMaxRelVCorr = 1.;
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
        boost::optional<double> tDact;
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
                if (!tDact) {
                    auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(this->geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
                    tDact.reset(1e-6 * tLeaf->getBoundingBox().height()); // m
                    #ifndef NDEBUG
                        this->writelog(LOG_DEBUG, "active layer thickness = %1%nm", 1e9 * *tDact);
                    #endif
                }
                double tJy = mCond[i].c11 * fabs(tDVy); // [j] = A/m²
                mHeatDensities[i] = phys::h_J * phys::c * tJy / ( phys::qe * real(inWavelength())*1e-9 * *tDact );
            } else if (this->geometry->getMaterial(tMidPoint)->kind() == Material::NONE || tRoles.find("noheat") != tRoles.end())
                mHeatDensities[i] = 0.;
            else
                mHeatDensities[i] = mCond[i].c00 * tDVx*tDVx + mCond[i].c11 * tDVy*tDVy;
        }
    }
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
    return interpolate(*((this->mesh)->getMidpointsMesh()), mCurrentDensities, WrappedMesh<2>(dst_mesh, this->geometry), method);
}

template<typename Geometry2DType> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities(const MeshD<2>& dst_mesh, InterpolationMethod method)
{
    if (!mPotentials) throw NoValue("Heat densities");
    this->writelog(LOG_DETAIL, "Getting heat densities");
    if (!mHeatDensities) saveHeatDensities(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mHeatDensities, WrappedMesh<2>(dst_mesh, this->geometry), method);
}


template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Beta2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.BetaCyl"; }

template struct FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
