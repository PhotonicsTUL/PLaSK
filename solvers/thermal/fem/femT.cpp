#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

template<typename Geometry2DType> FiniteElementMethodThermal2DSolver<Geometry2DType>::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectilinearMesh2D>(name),
    mBigNum(1e15),
    mCorrType(CORRECTION_ABSOLUTE),
    mTCorrLim(0.1),
    mTInit(300.),
    mLoopNo(0),
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
    this->writelog(LOG_DETAIL, "Setting up matrix system");

    auto iMesh = (this->mesh)->getMidpointsMesh();
    auto tHeatDensities = inHeatDensity(iMesh);

    std::fill_n(oA.data, mABand*mAOrder, 0.); // zero the matrix
    oLoad.fill(0.);

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

    // Set stiffness matrix and load vector
    for (auto ttE = mesh->elements.begin(); ttE != mesh->elements.end(); ++ttE)
    {
        // nodes numbers for the current element
        size_t tLoLeftNo = ttE->getLoLoIndex();
        size_t tLoRghtNo = ttE->getUpLoIndex();
        size_t tUpLeftNo = ttE->getLoUpIndex();
        size_t tUpRghtNo = ttE->getUpUpIndex();

        // element size
        double tElemWidth = ttE->getUpper0() - ttE->getLower0();
        double tElemHeight = ttE->getUpper1() - ttE->getLower1();

        // point and material in the middle of the element
        Vec<2,double> tMidPoint = ttE->getMidpoint();
        auto tMaterial = geometry->getMaterial(tMidPoint);

        // height of the leaf spanning over the element
        double tLayerHeight = dynamic_pointer_cast<const GeometryObjectD<2>>( geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf) )
                                -> getBoundingBox().sizeUp();

        // average temperature on the element
        double tTemp = 0.25 * (mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo] + mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]);

        // thermal conductivity
        double tKx, tKy;
        std::tie(tKx,tKy) = tMaterial->thermCond(tTemp, tLayerHeight);

        tKx *= tElemHeight; tKx /= tElemWidth;
        tKy *= tElemWidth; tKy /= tElemHeight;

        // load vector: heat densities
        double tF = 0.25e-12 * tElemWidth * tElemHeight * tHeatDensities[ttE->getIndex()]; // 1e-12 -> to transform um*um into m*m

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
    double* tAend = oA.data + oA.order * oA.band1;
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
//     this->writelog(LOG_INFO, "Setting up matrix system");
//
//     std::vector<Element2D>::const_iterator ttE = mElements.begin();
//     std::vector<Node2D>::const_iterator ttN = mNodes.begin();
//
//     size_t tLoLeftNo, tLoRghtNo, tUpLeftNo, tUpRghtNo, // nodes numbers in current element
//            tFstArg, tSecArg; // assistant values
//
//     double tKXAssist, tKYAssist, tElemWidth, tElemHeight, tF, // assistant values to set stiffness matrix
//            tK11, tK21, tK31, tK41, tK22, tK32, tK42, tK33, tK43, tK44, // local symetric matrix components
//            tF1hfX = 0., tF2hfX = 0., tF3hfX = 0., tF4hfX = 0., // for load vector (heat flux components for x-direction)
//            tF1hfY = 0., tF2hfY = 0., tF3hfY = 0., tF4hfY = 0., // for load vector (heat flux components for y-direction)
//            tK11convX = 0., tK21convX = 0., tK31convX = 0., tK41convX = 0., tK22convX = 0., // for symetric matrix (convection components for x-direction)
//            tK32convX = 0., tK42convX = 0., tK33convX = 0., tK43convX = 0., tK44convX = 0., // for symetric matrix (convection components for x-direction)
//            tK11convY = 0., tK21convY = 0., tK31convY = 0., tK41convY = 0., tK22convY = 0., // for symetric matrix (convection components for y-direction)
//            tK32convY = 0., tK42convY = 0., tK33convY = 0., tK43convY = 0., tK44convY = 0., // for symetric matrix (convection components for y-direction)
//            tF1convX = 0., tF2convX = 0., tF3convX = 0., tF4convX = 0., // for load vector (convection components for x-direction)
//            tF1convY = 0., tF2convY = 0., tF3convY = 0., tF4convY = 0., // for load vector (convection components for y-direction)
//            tF1radX = 0., tF2radX = 0., tF3radX = 0., tF4radX = 0., // for load vector (radiation components for x-direction)
//            tF1radY = 0., tF2radY = 0., tF3radY = 0., tF4radY = 0.; // for load vector (radiation components for y-direction)
//
//     // set zeros
//     for(int i = 0; i < mABand; i++)
//         for(int j = 0; j < mAOrder; j++)
//             mpA[i][j]=0.;
//
//     // set stiffness matrix and load vector
//     for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
//     {
//         // set nodes numbers for the current element
//         tLoLeftNo = ttE->getNLoLeftPtr()->getNo();
//         tLoRghtNo = ttE->getNLoRightPtr()->getNo();
//         tUpLeftNo = ttE->getNUpLeftPtr()->getNo();
//         tUpRghtNo = ttE->getNUpRightPtr()->getNo();
//
//         // set element size
//         tElemWidth = fabs(ttE->getNLoLeftPtr()->getX() - ttE->getNLoRightPtr()->getX());
//         tElemHeight = fabs(ttE->getNLoLeftPtr()->getY() - ttE->getNUpLeftPtr()->getY());
//
//         // set assistant values
//         std::vector<Box2D> tVecBox = (this->geometry)->getLeafsBoundingBoxes(); // geometry->extract(GeometryObject::PredicateHasClass("active"));
//         Vec<2, double> tSize;
//         for (Box2D tBox: tVecBox)
//         {
//             if (tBox.includes(vec(ttE->getX(), ttE->getY())))
//             {
//                 tSize = tBox.size();
//                 break;
//             }
//         }
//         tKXAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first;
//         tKYAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second;
//
//         // set load vector: heat densities
//         tF = 0.25 * tElemWidth * tElemHeight * 1e-12 * mHeatDensities[ttE->getNo()-1]; // 1e-12 -> to transform um*um into m*m
//
//         // boundary condition: heat flux
//         if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNLoRightPtr()->ifHFConst() ) // heat flux on bottom edge of the element
//         {
//             tF1hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getHF(); // 1e-6 -> to transform um into m
//             tF2hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getHF();
//         }
//         if ( ttE->getNUpLeftPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on top edge of the element
//         {
//             tF3hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getHF();
//             tF4hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getHF();
//         }
//         if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNUpLeftPtr()->ifHFConst() ) // heat flux on left edge of the element
//         {
//             tF1hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getHF();
//             tF4hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getHF();
//         }
//         if ( ttE->getNLoRightPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on right edge of the element
//         {
//             tF2hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getHF();
//             tF3hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getHF();
//         }
//
//         // boundary condition: convection
//         if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNLoRightPtr()->ifConvection() ) // convection on bottom edge of the element
//         {
//             tF1convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1(); // 1e-6 -> to transform um into m
//             tF2convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
//             tK22convX = tK11convX = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNLoRightPtr()->getConvCoeff()) * tElemWidth / 3.;
//             tK21convX = 0.5 * tK22convX;
//         }
//         if ( ttE->getNUpLeftPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on top edge of the element
//         {
//             tF3convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
//             tF4convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
//             tK44convX = tK33convX = (ttE->getNUpLeftPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemWidth / 3.;
//             tK43convX = 0.5 * tK44convX;
//         }
//         if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNUpLeftPtr()->ifConvection() ) // convection on left edge of the element
//         {
//             tF1convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1();
//             tF4convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
//             tK44convY = tK11convY = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNUpLeftPtr()->getConvCoeff()) * tElemHeight / 3.;
//             tK41convY = 0.5 * tK44convY;
//         }
//         if ( ttE->getNLoRightPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on right edge of the element
//         {
//             tF2convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
//             tF3convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
//             tK33convY = tK22convY = (ttE->getNLoRightPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemHeight / 3.;
//             tK32convY = 0.5 * tK33convY;
//         }
//
//         // boundary condition: radiation
//         if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNLoRightPtr()->ifRadiation() ) // radiation on bottom edge of the element
//         {
//             tF1radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * phys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4)); // 1e-6 -> to transform um into m
//             tF2radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * phys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
//         }
//         if ( ttE->getNUpLeftPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on top edge of the element
//         {
//             tF3radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * phys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
//             tF4radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * phys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
//         }
//         if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNUpLeftPtr()->ifRadiation() ) // radiation on left edge of the element
//         {
//             tF1radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * phys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4));
//             tF4radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * phys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
//         }
//         if ( ttE->getNLoRightPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on right edge of the element
//         {
//             tF2radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * phys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
//             tF3radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * phys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
//         }
//
//         // set symetric matrix components
//         tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
//         tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
//         tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
//         tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;
//
//         // set stiffness matrix
//         mpA[tLoLeftNo-1][mAOrder-2] += ( ttE->getX() * (tK11 + tK11convX + tK11convY) );
//         mpA[tLoRghtNo-1][mAOrder-2] += ( ttE->getX() * (tK22 + tK22convX + tK22convY) );
//         mpA[tUpRghtNo-1][mAOrder-2] += ( ttE->getX() * (tK33 + tK33convX + tK33convY) );
//         mpA[tUpLeftNo-1][mAOrder-2] += ( ttE->getX() * (tK44 + tK44convX + tK44convY) );
//
//         tLoRghtNo > tLoLeftNo ? (tFstArg = tLoRghtNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tLoRghtNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK21 + tK21convX + tK21convY) );
//
//         tUpRghtNo > tLoLeftNo ? (tFstArg = tUpRghtNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpRghtNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK31 + tK31convX + tK31convY) );
//
//         tUpLeftNo > tLoLeftNo ? (tFstArg = tUpLeftNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpLeftNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK41 + tK41convX + tK41convY) );
//
//         tUpRghtNo > tLoRghtNo ? (tFstArg = tUpRghtNo, tSecArg = tLoRghtNo) : (tFstArg = tLoRghtNo, tSecArg = tUpRghtNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK32 + tK32convX + tK32convY) );
//
//         tUpLeftNo > tLoRghtNo ? (tFstArg = tUpLeftNo, tSecArg = tLoRghtNo) : (tFstArg = tLoRghtNo, tSecArg = tUpLeftNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK42 + tK42convX + tK42convY) );
//
//         tUpLeftNo > tUpRghtNo ? (tFstArg = tUpLeftNo, tSecArg = tUpRghtNo) : (tFstArg = tUpRghtNo, tSecArg = tUpLeftNo);
//         mpA[tFstArg-1][mAOrder-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK43 + tK43convX + tK43convY) );
//
//         // set load vector
//         mpA[tLoLeftNo-1][mAOrder-1]  += ( ttE->getX() * (tF + tF1hfX + tF1hfY + tF1convX + tF1convY + tF1radX + tF1radY) );
//         mpA[tLoRghtNo-1][mAOrder-1] += ( ttE->getX() * (tF + tF2hfX + tF2hfY + tF2convX + tF2convY + tF2radX + tF2radY) );
//         mpA[tUpRghtNo-1][mAOrder-1] += ( ttE->getX() * (tF + tF3hfX + tF3hfY + tF3convX + tF3convY + tF3radX + tF3radY) );
//         mpA[tUpLeftNo-1][mAOrder-1]  += ( ttE->getX() * (tF + tF4hfX + tF4hfY + tF4convX + tF4convY + tF4radX + tF4radY) );
//     }
//     // boundary conditions are taken into account
//     for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
//         if ( ttN->ifTConst() )
//         {
//             mpA[ttN->getNo()-1][mAOrder-2] += mBigNum;
//             mpA[ttN->getNo()-1][mAOrder-1] += ttN->getT()*mBigNum;
//         }
}


template<typename Geometry2DType> double FiniteElementMethodThermal2DSolver<Geometry2DType>::calculate(int iLoopLim)
{
    this->initCalculation();

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

    // mTemperatures = mTemperatures.claim(); // FIXME should we call it or not? this is safer if someone else holds our temperature, but reduces performance!
    DataVector<double> tT(mAOrder);

    do {
        ++mLoopNo;
        ++tLoop;

        setMatrix(tA, tT, tTConst, tHFConst, tConvection, tRadiation);

        int tInfo = solveMatrix(tA, tT);

        if (tInfo > 0)
            throw ComputationError(this->getId(), "Leading minor of order %1% of the stiffness matrix is not positive-definite", tInfo);

        saveTemperatures(tT);

        if (mMaxAbsTCorr > tMaxMaxAbsTCorr) tMaxMaxAbsTCorr = mMaxAbsTCorr;
        if (mMaxRelTCorr > tMaxMaxRelTCorr) tMaxMaxRelTCorr = mMaxRelTCorr;

        // show max correction
        this->writelog(LOG_DATA, "Loop no: %d(%d), max. T update: %8.6f (%8.6f%%)", tLoop, mLoopNo, mMaxAbsTCorr, 100.*mMaxRelTCorr);

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
                else throw XMLBadAttrException(source, "algorithm", *tAlgo, + "\"block\" or \"slow\"");
            }
            source.requireTagEnd();
        } else
            this->parseStandardConfiguration(source, manager);
    }
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::saveTemperatures(DataVector<double>& iT)
{
    mMaxAbsTCorr = 0.;
    mMaxRelTCorr = 0.;

    for (auto ttTemp = mTemperatures.begin(), ttT = iT.begin(); ttT != iT.end(); ++ttTemp, ++ttT)
    {
        double tAbsCorr = std::abs(*ttT - *ttTemp); // for boundary with constant temperature this will be zero anyway
        double tRelCorr = tAbsCorr / *ttT;
        if (tAbsCorr > mMaxAbsTCorr) mMaxAbsTCorr = tAbsCorr;
        if (tRelCorr > mMaxRelTCorr) mMaxRelTCorr = tRelCorr;
    }
    std::swap(mTemperatures, iT);
}


template<typename Geometry2DType> void FiniteElementMethodThermal2DSolver<Geometry2DType>::saveHeatFluxes()
{
    this->writelog(LOG_INFO, "Computing heat fluxes");

    mHeatFluxes.reset(this->mesh->elements.size());

    for (auto ttE = this->mesh->elements.begin(); ttE != this->mesh->elements.end(); ++ttE)
    {
        Vec<2,double> tMidPoint = ttE->getMidpoint();
        auto tMaterial = this->geometry->getMaterial(tMidPoint);

        double tLayerHeight =
            dynamic_pointer_cast<const GeometryObjectD<2>>( this->geometry->getMatchingAt(tMidPoint, &GeometryObject::PredicateIsLeaf) )
                                -> getBoundingBox().sizeUp();

        size_t tLoLeftNo = ttE->getLoLoIndex();
        size_t tLoRghtNo = ttE->getUpLoIndex();
        size_t tUpLeftNo = ttE->getLoUpIndex();
        size_t tUpRghtNo = ttE->getUpUpIndex();

        double tKx, tKy;
        std::tie(tKx,tKy) = tMaterial->thermCond(0.25 * (mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo] +
                                                         mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]  ), tLayerHeight);

        mHeatFluxes[ttE->getIndex()] = vec(
            - 0.5e6 * tKx * (- mTemperatures[tLoLeftNo] + mTemperatures[tLoRghtNo]
                             - mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo]), // 1e6 - from um to m
            - 0.5e6 * tKy * (- mTemperatures[tLoLeftNo] - mTemperatures[tLoRghtNo]
                             + mTemperatures[tUpLeftNo] + mTemperatures[tUpRghtNo])); // 1e6 - from um to m
    }
}


template<typename Geometry2DType> int FiniteElementMethodThermal2DSolver<Geometry2DType>::solveMatrix(BandSymMatrix& iA, DataVector<double>& ioB)
{
    this->writelog(LOG_DETAIL, "Solving matrix system");

    int info = 0;

    // Factorize matrix
    switch (mAlgorithm) {
        case ALGORITHM_SLOW:
            dpbtf2(UPLO, iA.order, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtf2 has illegal value", this->getId(), -info);
            break;
        case ALGORITHM_BLOCK:
            dpbtrf(UPLO, iA.order, iA.band1, iA.data, iA.band1+1, info);
            if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrf has illegal value", this->getId(), -info);
            break;
    }
    if (info > 0) return info;

    // Find solutions
    dpbtrs(UPLO, iA.order, iA.band1, 1, iA.data, iA.band1+1, ioB.data(), ioB.size(), info);
    if (info < 0) throw CriticalException("%1%: Argument %2% of dpbtrs has illegal value", this->getId(), -info);

    // now iA contains factorized matrix and ioB the solutions
    return info;
}


template<typename Geometry2DType> DataVector<const double> FiniteElementMethodThermal2DSolver<Geometry2DType>::getTemperatures(const MeshD<2> &dst_mesh, InterpolationMethod method) const {
    if (!mTemperatures) return DataVector<const double>(dst_mesh.size(), mTInit); // in case the receiver is connected and no temperature calculated yet
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mTemperatures, dst_mesh, method);
}


template<typename Geometry2DType> DataVector<const Vec<2> > FiniteElementMethodThermal2DSolver<Geometry2DType>::getHeatFluxes(const MeshD<2> &dst_mesh, InterpolationMethod method) {
    if (!mTemperatures) return DataVector<const Vec<2>>(dst_mesh.size(), Vec<2>(0.,0.)); // in case the receiver is connected and no fluxes calculated yet
    if (!mHeatFluxes) saveHeatFluxes(); // we will compute fluxes only if they are needed
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mHeatFluxes, dst_mesh, method);
}


template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCartesian>::getClassName() const { return "thermal.Fem2D"; }
template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>::getClassName() const { return "thermal.FemCyl"; }

template struct FiniteElementMethodThermal2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
