#include "femT.h"
#include "dcg.h"

namespace plask { namespace solvers { namespace thermal {

template<typename Geometry2DType> FiniteElementMethodThermal2DSolver<Geometry2DType>::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    mBigNum(1e15),
    mTCorrLim(0.05),
    mTInit(300.),
    mLoopNo(0),
    mCorrType(CORRECTION_ABSOLUTE),
    outTemperature(this, &FiniteElementMethodThermal2DSolver<Geometry2DType>::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodThermal2DSolver<Geometry2DType>::getHeatFluxes),
    mAlgorithm(ALGORITHM_BLOCK)
{
    mTemperatures.reset();
    mHeatFluxes.reset();
    inHeatDensity = 0.;
}


template<typename Geometry2DType> FiniteElementMethodThermal2DSolver<Geometry2DType>::~FiniteElementMethodThermal2DSolver() {
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();

        if (param == "temperature")
            this->readBoundaryConditions(manager, source, mTConst);

        else if (param == "heatflux")
            this->readBoundaryConditions(manager, source, mHFConst);

        else if (param == "convection")
            this->readBoundaryConditions(manager, source, mConvection);

        else if (param == "radiation")
            this->readBoundaryConditions(manager, source, mRadiation);

        else if (param == "loop") {
            mTInit = source.getAttribute<double>("inittemp", mTInit);
            mTCorrLim = source.getAttribute<double>("corrlim", mTCorrLim);
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
                //else throw XMLBadAttrException(source, "algorithm", *tAlgo, + "\"block\", \"slow\", or \"iterative\"");
                else throw XMLBadAttrException(source, "algorithm", *tAlgo, + "\"block\" or \"slow\"");
            }
            source.requireTagEnd();
        } else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::onInitialize() {
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    mLoopNo = 0;
    mAOrder = this->mesh->size();
    mABand = this->mesh->minorAxis().size() + 2;
    mTemperatures.reset(mAOrder, mTInit);
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::onInvalidate() {
    mTemperatures.reset();
    mHeatFluxes.reset();
}


/**
    * Helper function for applying boundary conditions of element edges to stiffness matrix.
    * Boundary conditions must be set for both nodes at the element edge.
    * \param boundary_conditions boundary conditions holder
    * \param i1, i2, i3, i4 indices of the lower left, lower right, upper right, and upper left node
    * \param width width of the element
    * \param height height of the element
    * \param[out] F1, F2, F3, F4 references to the load vector components
    * \param[out] K11, K22, K33, K44, K12, K14, K24, K34 references to the stiffness matrix components
    * \param F_function function returning load vector component
    * \param Kmm_function function returning stiffness matrix diagonal component
    * \param Kmn_function function returning stiffness matrix off-diagonal component
    */
template <typename ConditionT>
static void setBoundaries(const BoundaryConditionsWithMesh<RectilinearMesh2D,ConditionT>& boundary_conditions,
                          size_t i1, size_t i2, size_t i3, size_t i4, double width, double height,
                          double& F1, double& F2, double& F3, double& F4,
                          double& K11, double& K22, double& K33, double& K44,
                          double& K12, double& K23, double& K34, double& K41,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& F_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& Kmm_function,
                          const std::function<double(double,ConditionT,ConditionT,size_t,size_t)>& Kmn_function
                         )
{
    auto val1 = boundary_conditions.getValue(i1);
    auto val2 = boundary_conditions.getValue(i2);
    auto val3 = boundary_conditions.getValue(i3);
    auto val4 = boundary_conditions.getValue(i4);
    if (val1 && val2) { // bottom
        F1 += F_function(width, *val1, *val2, i1, i2); F2 += F_function(width, *val2, *val1, i2, i1);
        K11 += Kmm_function(width, *val1, *val2, i1, i2); K22 += Kmm_function(width, *val2, *val1, i2, i1);
        K12 += Kmn_function(width, *val1, *val2, i1, i2);
    }
    if (val2 && val3) { // right
        F2 += F_function(height, *val2, *val3, i2, i3); F3 += F_function(height, *val3, *val2, i3, i2);
        K22 += Kmm_function(height, *val2, *val3, i2, i3); K33 += Kmm_function(height, *val3, *val2, i3, i2);
        K23 += Kmn_function(height, *val2, *val3, i2, i3);
    }
    if (val3 && val4) { // top
        F3 += F_function(width, *val3, *val4, i3, i4); F4 += F_function(width, *val4, *val3, i4, i3);
        K33 += Kmm_function(width, *val3, *val4, i3, i4); K44 += Kmm_function(width, *val4, *val3, i4, i3);
        K34 += Kmn_function(width, *val3, *val4, i3, i4);
    }
    if (val4 && val1) { // left
        F1 += F_function(height, *val1, *val4, i1, i4); F4 += F_function(height, *val4, *val1, i4, i1);
        K11 += Kmm_function(height, *val1, *val4, i1, i4); K44 += Kmm_function(height, *val4, *val1, i4, i1);
        K41 += Kmn_function(height, *val1, *val4, i1, i4);
    }
}




template<>
void FiniteElementMethodThermal2DSolver<Geometry2DCartesian>::setMatrix(BandSymMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iTConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iHFConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Convection>& iConvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Radiation>& iRadiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, band=%2%)", mAOrder, mABand);

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tHeatDensities = inHeatDensity(iMesh);

    std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
    oLoad.fill(0.);

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

    // Set stiffness matrix and load vector
    for (auto tE: this->mesh->elements)
    {
        // nodes numbers for the current element
        size_t tLoLeftNo = tE.getLoLoIndex();
        size_t tLoRghtNo = tE.getUpLoIndex();
        size_t tUpLeftNo = tE.getLoUpIndex();
        size_t tUpRghtNo = tE.getUpUpIndex();

        // element size
        double tElemWidth = tE.getUpper0() - tE.getLower0();
        double tElemHeight = tE.getUpper1() - tE.getLower1();

        // point and material in the middle of the element
        Vec<2,double> tMidPoint = tE.getMidpoint();
        auto tMaterial = geometry->getMaterial(tMidPoint);

        // average temperature on the element
        double tTemp = 0.25 * (mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo] + mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]);

        // thermal conductivity
        double tKx, tKy;
        auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
        if (tLeaf)
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp, tLeaf->getBoundingBox().height());
        else
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp);

        tKx *= tElemHeight; tKx /= tElemWidth;
        tKy *= tElemWidth; tKy /= tElemHeight;

        // load vector: heat densities
        double tF = 0.25e-12 * tElemWidth * tElemHeight * tHeatDensities[tE.getIndex()]; // 1e-12 -> to transform µm² into m²

        // set symmetric matrix components
        double tK44, tK33, tK22, tK11, tK43, tK21, tK42, tK31, tK32, tK41;

        tK44 = tK33 = tK22 = tK11 = (tKx + tKy) / 3.;
        tK43 = tK21 = (-2. * tKx + tKy) / 6.;
        tK42 = tK31 = - (tKx + tKy) / 6.;
        tK32 = tK41 = (tKx - 2. * tKy) / 6.;

        double tF1 = tF, tF2 = tF, tF3 = tF, tF4 = tF;

        // boundary conditions: heat flux
        setBoundaries<double>(iHFConst, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [](double len, double val, double, size_t, size_t) { // F
                          return - 0.5e-6 * len * val;
                      },
                      [](double,double,double,size_t,size_t){return 0.;}, // K diagonal
                      [](double,double,double,size_t,size_t){return 0.;}  // K off-diagonal
                     );

        // boundary conditions: convection
        setBoundaries<Convection>(iConvection, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [](double len, Convection val, Convection, size_t, size_t) { // F
                          return 0.5e-6 * len * val.mConvCoeff * val.mTAmb1;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t) { // K diagonal
                          return (val1.mConvCoeff + val2.mConvCoeff) * len / 3.;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t) { // K off-diagonal
                          return (val1.mConvCoeff + val2.mConvCoeff) * len / 6.;
                      }
                     );

        // boundary conditions: radiation
        setBoundaries<Radiation>(iRadiation, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [this](double len, Radiation val, Radiation, size_t i, size_t) -> double { // F
                          double a = val.mTAmb2; a = a*a;
                          double T = this->mTemperatures[i]; T = T*T;
                          return - 0.5e-6 * len * val.mSurfEmiss * phys::SB * (T*T - a*a);},
                      [](double,Radiation,Radiation,size_t,size_t){return 0.;}, // K diagonal
                      [](double,Radiation,Radiation,size_t,size_t){return 0.;}  // K off-diagonal
                     );

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

        // set load vector
        oLoad[tLoLeftNo] += tF1;
        oLoad[tLoRghtNo] += tF2;
        oLoad[tUpRghtNo] += tF3;
        oLoad[tUpLeftNo] += tF4;
    }

    // boundary conditions of the first kind
    for (auto tCond: iTConst) {
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


template<>
void FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>::setMatrix(BandSymMatrix& oA, DataVector<double>& oLoad,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iTConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,double>& iHFConst,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Convection>& iConvection,
                   const BoundaryConditionsWithMesh<RectilinearMesh2D,Radiation>& iRadiation
                  )
{
    this->writelog(LOG_DETAIL, "Setting up matrix system (size=%1%, band=%2%)", mAOrder, mABand);

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tHeatDensities = inHeatDensity(iMesh);

    std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
    oLoad.fill(0.);

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

    // Set stiffness matrix and load vector
    for (auto tE: this->mesh->elements)
    {
        // nodes numbers for the current element
        size_t tLoLeftNo = tE.getLoLoIndex();
        size_t tLoRghtNo = tE.getUpLoIndex();
        size_t tUpLeftNo = tE.getLoUpIndex();
        size_t tUpRghtNo = tE.getUpUpIndex();

        // element size
        double tElemWidth = tE.getUpper0() - tE.getLower0();
        double tElemHeight = tE.getUpper1() - tE.getLower1();

        // point and material in the middle of the element
        Vec<2,double> tMidPoint = tE.getMidpoint();
        auto tMaterial = geometry->getMaterial(tMidPoint);
        double r = tMidPoint.rad_r();

        // average temperature on the element
        double tTemp = 0.25 * (mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo] + mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]);

        // thermal conductivity
        double tKx, tKy;
        auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf));
        if (tLeaf)
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp, tLeaf->getBoundingBox().height());
        else
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp);

        tKx = tKx * tElemHeight / tElemWidth;
        tKy = tKy * tElemWidth / tElemHeight;

        // load vector: heat densities
        double tF = 0.25e-12 * tElemWidth * tElemHeight * tHeatDensities[tE.getIndex()]; // 1e-12 -> to transform um*um into m*m

        // set symmetric matrix components
        double tK44, tK33, tK22, tK11, tK43, tK21, tK42, tK31, tK32, tK41;

        tK44 = tK33 = tK22 = tK11 = (tKx + tKy) / 3.;
        tK43 = tK21 = (-2. * tKx + tKy) / 6.;
        tK42 = tK31 = - (tKx + tKy) / 6.;
        tK32 = tK41 = (tKx - 2. * tKy) / 6.;

        double tF1 = tF, tF2 = tF, tF3 = tF, tF4 = tF;

        // boundary conditions: heat flux
        setBoundaries<double>(iHFConst, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [](double len, double val, double, size_t, size_t) { // F
                          return - 0.5e-6 * len * val;
                      },
                      [](double,double,double,size_t,size_t){return 0.;}, // K diagonal
                      [](double,double,double,size_t,size_t){return 0.;}  // K off-diagonal
                     );

        // boundary conditions: convection
        setBoundaries<Convection>(iConvection, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [](double len, Convection val, Convection, size_t, size_t) { // F
                          return 0.5e-6 * len * val.mConvCoeff * val.mTAmb1;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t) { // K diagonal
                          return (val1.mConvCoeff + val2.mConvCoeff) * len / 3.;
                      },
                      [](double len, Convection val1, Convection val2, size_t, size_t) { // K off-diagonal
                          return (val1.mConvCoeff + val2.mConvCoeff) * len / 6.;
                      }
                     );

        // boundary conditions: radiation
        setBoundaries<Radiation>(iRadiation, tLoLeftNo, tLoRghtNo, tUpRghtNo, tUpLeftNo, tElemWidth, tElemHeight,
                      tF1, tF2, tF3, tF4, tK11, tK22, tK33, tK44, tK21, tK32, tK43, tK41,
                      [this](double len, Radiation val, Radiation, size_t i, size_t) -> double { // F
                          double a = val.mTAmb2; a = a*a;
                          double T = this->mTemperatures[i]; T = T*T;
                          return - 0.5e-6 * len * val.mSurfEmiss * phys::SB * (T*T - a*a);},
                      [](double,Radiation,Radiation,size_t,size_t){return 0.;}, // K diagonal
                      [](double,Radiation,Radiation,size_t,size_t){return 0.;}  // K off-diagonal
                     );

        // set stiffness matrix
        oA(tLoLeftNo, tLoLeftNo) += r * tK11;
        oA(tLoRghtNo, tLoRghtNo) += r * tK22;
        oA(tUpRghtNo, tUpRghtNo) += r * tK33;
        oA(tUpLeftNo, tUpLeftNo) += r * tK44;

        oA(tLoRghtNo, tLoLeftNo) += r * tK21;
        oA(tUpRghtNo, tLoLeftNo) += r * tK31;
        oA(tUpLeftNo, tLoLeftNo) += r * tK41;
        oA(tUpRghtNo, tLoRghtNo) += r * tK32;
        oA(tUpLeftNo, tLoRghtNo) += r * tK42;
        oA(tUpLeftNo, tUpRghtNo) += r * tK43;

        // set load vector
        oLoad[tLoLeftNo] += r * tF1;
        oLoad[tLoRghtNo] += r * tF2;
        oLoad[tUpRghtNo] += r * tF3;
        oLoad[tUpLeftNo] += r * tF4;
    }

    // boundary conditions of the first kind
    for (auto tCond: iTConst) {
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


template<typename Geometry2DType> double FiniteElementMethodThermal2DSolver<Geometry2DType>::compute(int iLoopLim)
{
    this->initCalculation();

    mHeatFluxes.reset();

    // store boundary conditions for current mesh
    auto tTConst = mTConst(this->mesh);
    auto tHFConst = mHFConst(this->mesh);
    auto tConvection = mConvection(this->mesh);
    auto tRadiation = mRadiation(this->mesh);

    this->writelog(LOG_INFO, "Running thermal calculations");

    int tLoop = 0;
    BandSymMatrix tA(mAOrder, mABand);

    double tMaxMaxAbsTCorr = 0.,
           tMaxMaxRelTCorr = 0.;

#   ifndef NDEBUG
        if (!mTemperatures.unique()) this->writelog(LOG_DEBUG, "Temperature data held by something else...");
#   endif
    mTemperatures = mTemperatures.claim(); // FIXME should we call it or not? this is safer if someone else holds our temperature, but reduces performance!
    DataVector<double> tT(mAOrder);

    do {
        setMatrix(tA, tT, tTConst, tHFConst, tConvection, tRadiation);

        solveMatrix(tA, tT);

        saveTemperatures(tT);

        if (mMaxAbsTCorr > tMaxMaxAbsTCorr) tMaxMaxAbsTCorr = mMaxAbsTCorr;
        if (mMaxRelTCorr > tMaxMaxRelTCorr) tMaxMaxRelTCorr = mMaxRelTCorr;

        ++mLoopNo;
        ++tLoop;

        // show max correction
        this->writelog(LOG_RESULT, "Loop %d(%d): max(T)=%.3fK, update=%.3fK(%.3f%%)", tLoop, mLoopNo, mMaxT, mMaxAbsTCorr, mMaxRelTCorr);

    } while (((mCorrType == CORRECTION_ABSOLUTE)? (mMaxAbsTCorr > mTCorrLim) : (mMaxRelTCorr > mTCorrLim)) && (iLoopLim == 0 || tLoop < iLoopLim));

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    // Make sure we store the maximum encountered values, not just the last ones
    // (so, this will indicate if the results changed since the last run, not since the last loop iteration)
    mMaxAbsTCorr = tMaxMaxAbsTCorr;
    mMaxRelTCorr = tMaxMaxRelTCorr;

    if (mCorrType == CORRECTION_RELATIVE) return mMaxRelTCorr;
    else return mMaxAbsTCorr;
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::solveMatrix(BandSymMatrix& iA, DataVector<double>& ioB)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    switch (mAlgorithm) {

        case ALGORITHM_SLOW:
            //Factorize
            dpbtf2(UPLO, iA.size, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0)
                throw CriticalException("%1%: Argument %2% of dpbtf2 has illegal value", this->getId(), -info);
            else if (info > 0)
                throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);
            // Find solutions
            dpbtrs(UPLO, iA.size, iA.band1, 1, iA.data, iA.band1+1, ioB.data(), ioB.size(), info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);
            break;

        case ALGORITHM_BLOCK:
            //Factorize
            dpbtrf(UPLO, iA.size, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0)
                throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
            else if (info > 0)
                throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", info);
            // Find solutions
            dpbtrs(UPLO, iA.size, iA.band1, 1, iA.data, iA.band1+1, ioB.data(), ioB.size(), info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);
            break;

        case ALGORITHM_ITERATIVE:
            AtimesDSB atimes(iA);
            MsolveJacobiDSB msolve(iA);
            DataVector<double> oX = mTemperatures.copy(); // We use previous temperatures as initial solution
            double err;
            try {
                int iter = solveDCG(iA.size, atimes, msolve, oX.data(), ioB.data(), err); //TODO add parameters for tolerance and maximum iterations
                this->writelog(LOG_DETAIL, "Conjugate gradient converged after %1% iterations.", iter);
            } catch (DCGError err) {
                throw ComputationError(this->getId(), "Conjugate gradient failed:, %1%", err.what());
            }
            ioB = oX;

            break;
    }

    // now iA contains factorized matrix and ioB the solutions
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::saveTemperatures(DataVector<double>& iT)
{
    mMaxAbsTCorr = 0.;
    mMaxRelTCorr = 0.;

    mMaxT = 0.;

    for (auto ttTemp = mTemperatures.begin(), ttT = iT.begin(); ttT != iT.end(); ++ttTemp, ++ttT)
    {
        double tAbsCorr = std::abs(*ttT - *ttTemp); // for boundary with constant temperature this will be zero anyway
        double tRelCorr = tAbsCorr / *ttT;
        if (tAbsCorr > mMaxAbsTCorr) mMaxAbsTCorr = tAbsCorr;
        if (tRelCorr > mMaxRelTCorr) mMaxRelTCorr = tRelCorr;
        if (*ttT > mMaxT) mMaxT = *ttT;
    }
    mMaxRelTCorr *= 100.; // %
    std::swap(mTemperatures, iT);
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::saveHeatFluxes()
{
    this->writelog(LOG_DETAIL, "Computing heat fluxes");

    mHeatFluxes.reset(this->mesh->elements.size());

    for (auto tE: this->mesh->elements)
    {
        Vec<2,double> tMidPoint = tE.getMidpoint();
        auto tMaterial = this->geometry->getMaterial(tMidPoint);

        size_t tLoLeftNo = tE.getLoLoIndex();
        size_t tLoRghtNo = tE.getUpLoIndex();
        size_t tUpLeftNo = tE.getLoUpIndex();
        size_t tUpRghtNo = tE.getUpUpIndex();

        double tTemp = 0.25 * (mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo] +
                               mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]);

        double tKx, tKy;
        auto tLeaf = dynamic_pointer_cast<const GeometryObjectD<2>>(
                        this->geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf)
                     );
        if (tLeaf)
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp, tLeaf->getBoundingBox().height());
        else
            std::tie(tKx,tKy) = tMaterial->thermk(tTemp);


        mHeatFluxes[tE.getIndex()] = vec(
            - 0.5e6 * tKx * (- mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo]
                             - mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]) / (tE.getUpper0() - tE.getLower0()), // 1e6 - from um to m
            - 0.5e6 * tKy * (- mTemperatures[tLoLeftNo] - mTemperatures[tLoRghtNo]
                             + mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]) / (tE.getUpper1() - tE.getLower1())); // 1e6 - from um to m
    }
}


template<typename Geometry2DType> DataVector<const double> FiniteElementMethodThermal2DSolver<Geometry2DType>::getTemperatures(const MeshD<2>& dst_mesh, InterpolationMethod method) const {
    this->writelog(LOG_DETAIL, "Getting temperatures");
    if (!mTemperatures) return DataVector<const double>(dst_mesh.size(), mTInit); // in case the receiver is connected and no temperature calculated yet
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mTemperatures, WrappedMesh<2>(dst_mesh, this->geometry), method);
}


template<typename Geometry2DType> DataVector<const Vec<2> > FiniteElementMethodThermal2DSolver<Geometry2DType>::getHeatFluxes(const MeshD<2>& dst_mesh, InterpolationMethod method) {
    this->writelog(LOG_DETAIL, "Getting heat fluxes");
    if (!mTemperatures) return DataVector<const Vec<2>>(dst_mesh.size(), Vec<2>(0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!mHeatFluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mHeatFluxes, WrappedMesh<2>(dst_mesh, this->geometry), method);
}


template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCartesian>::getClassName() const { return "thermal.Rectangular2D"; }
template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>::getClassName() const { return "thermal.RectangularCyl"; }

template struct FiniteElementMethodThermal2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
