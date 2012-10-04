#include "femT.h"


namespace plask { namespace solvers { namespace thermal {

template<typename Geometry2Dtype> FiniteElementMethodThermal2DSolver<Geometry2Dtype>::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2Dtype, RectilinearMesh2D>(name),
    mLoopLim(5),
    mTCorrLim(0.1),
    mTBigCorr(1e5),
    mBigNum(1e15),
    mTInit(300.),
    mLogs(false),
    mLoopNo(0),
    outTemperature(this, &FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getHeatFluxes)
{
    mNodes.clear();
    mElements.clear();
    mTemperatures.reset();
    mHeatFluxes.reset();
    mHeatDensities.reset();

    inHeatDensity = 0.;

}

template<typename Geometry2Dtype> FiniteElementMethodThermal2DSolver<Geometry2Dtype>::~FiniteElementMethodThermal2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::onInitialize() { // In this function check if geometry and mesh are set
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    this->setNodes();
    this->setElements();
    mTemperatures.reset(mNodes.size(), mTInit);
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    mNodes.clear();
    mElements.clear();
    mTemperatures.reset();
    mHeatFluxes.reset();
    mHeatDensities.reset();
    // Make sure that no provider returns any value.
    // If this method has been called, before next computations, onInitialize will be called.
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setLoopLim(int iLoopLim) { mLoopLim = iLoopLim; }
template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setTCorrLim(double iTCorrLim) { mTCorrLim = iTCorrLim; }
template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setTBigCorr(double iTBigCorr) { mTBigCorr = iTBigCorr; }
template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setBigNum(double iBigNum) { mBigNum = iBigNum; }
template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setTInit(double iTInit) { mTInit = iTInit; }

template<typename Geometry2Dtype> int FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getLoopLim() { return mLoopLim; }
template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getTCorrLim() { return mTCorrLim; }
template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getTBigCorr() { return mTBigCorr; }
template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getBigNum() { return mBigNum; }
template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getTInit() { return mTInit; }

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setNodes()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting nodes...");

    size_t tNo = 1; // node number

    Node2D* tpN = NULL;

    for(plask::RectilinearMesh2D::iterator vec_it = (this->mesh)->begin(); vec_it != (this->mesh)->end(); ++vec_it) // loop through all nodes given in the correct iteration order
    {
        double x = vec_it->ee_x();
        double y = vec_it->ee_y();

        std::size_t i = vec_it.getIndex();

        // checking boundary condition - constant temperature
        auto it1 = mTConst.includes(*(this->mesh), i);
        if (it1 != mTConst.end())
            tpN = new Node2D(tNo, x, y, it1->condition, true);
        else
            tpN = new Node2D(tNo, x, y, mTInit, false);

        // checking boundary condition - constant heat flux
        auto it2 = mHFConst.includes(*(this->mesh), i);
        if (it2 != mHFConst.end())
        {
            tpN->setHF(it2->condition);
            tpN->setHFflag(true);
        }
        else
            tpN->setHFflag(false);

        // checking boundary condition - convection
        auto it3 = mConvection.includes(*(this->mesh), i);
        if (it3 != mConvection.end())
        {
            tpN->setConv(it3->condition.mConvCoeff, it3->condition.mTAmb1);
            tpN->setConvflag(true);
        }
        else
            tpN->setConvflag(false);

        // checking boundary condition - radiation
        auto it4 = mRadiation.includes(*(this->mesh), i);
        if (it4 != mRadiation.end())
        {
            tpN->setRad(it4->condition.mSurfEmiss, it4->condition.mTAmb2);
            tpN->setRadflag(true);
        }
        else
            tpN->setRadflag(false);

        mNodes.push_back(*tpN);

        delete tpN;
        tNo++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setElements()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting elements...");

    size_t tNo = 1; // element number

    Element2D* tpE = NULL;

    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    for(std::size_t i = 0; i < (this->mesh)->majorAxis().size()-1; i++)
    {
        for(std::size_t j = 0; j < (this->mesh)->minorAxis().size()-1; j++)
        {
            if ((this->mesh)->getIterationOrder() == 0) // more nodes in y-direction
                tpE = new Element2D(tNo, &(*ttN), &(*(ttN+1)), &(*(ttN+((this->mesh)->minorAxis().size()))), &(*(ttN+((this->mesh)->minorAxis().size()+1))));
            else // (mesh->getIterationOrder() == 1)
                tpE = new Element2D(tNo, &(*ttN), &(*(ttN+((this->mesh)->minorAxis().size()))), &(*(ttN+1)), &(*(ttN+((this->mesh)->minorAxis().size()+1))));
            tpE->setT();
            mElements.push_back(*tpE);

            delete tpE;
            tNo++;
            ttN++;
        }
        ttN++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setHeatDensities()
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    try
    {
        mHeatDensities = inHeatDensity(iMesh);
    }
    catch (NoValue)
    {
        mHeatDensities.reset(iMesh->size(), 0.);
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::setSolver()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting solver...");

    /*size_t tNoOfNodesX, tNoOfNodesY; // TEST

    if (mesh->getIterationOrder() == 0) // 0: fast x, slow y
    {
        tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
        tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (larger one)
    }
    else // mesh->getIterationOrder() == 1 // 1: fast y, slow x
    {
        tNoOfNodesY = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
        tNoOfNodesX = mesh->majorAxis().size(); // number of nodes on major axis (larger one)
    };

    std::cout << "Number of nodes in x-direction: " << tNoOfNodesX << std::endl; // TEST
    std::cout << "Number of nodes in y-direction: " << tNoOfNodesY << std::endl; // TEST

    size_t tNoOfNodes = mesh->size(); // number of all nodes // TEST
    std::cout << "Number of nodes: " << tNoOfNodes << std::endl;*/ // TEST

    mAWidth = (this->mesh)->minorAxis().size() + 3;
    mAHeight = mNodes.size();
    mpA = new double*[mAHeight];
    for(int i = 0; i < mAHeight; i++)
        mpA[i] = new double[mAWidth];
    mTCorr.clear();
    for(int i = 0; i < mAHeight; i++)
        mTCorr.push_back(mTBigCorr);

    //std::cout << "Main matrix width: " << mAWidth << std::endl; // TEST
    //std::cout << "Main matrix height: " << mAHeight << std::endl; // TEST
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::delSolver()
{
    if (mLogs)
        writelog(LOG_INFO, "Deleting solver...");

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mTCorr.clear();
}

template<> void FiniteElementMethodThermal2DSolver<Geometry2DCartesian>::setMatrix()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting matrix...");

    std::vector<Element2D>::const_iterator ttE = mElements.begin();
    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    size_t tLoLeftNo, tLoRightNo, tUpLeftNo, tUpRightNo, // nodes numbers in current element
           tFstArg, tSecArg; // assistant values

    double tKXAssist, tKYAssist, tElemWidth, tElemHeight, tF, // assistant values to set stiffness matrix
           tK11, tK21, tK31, tK41, tK22, tK32, tK42, tK33, tK43, tK44, // local symetric matrix components
           tF1hfX = 0., tF2hfX = 0., tF3hfX = 0., tF4hfX = 0., // for load vector (heat flux components for x-direction)
           tF1hfY = 0., tF2hfY = 0., tF3hfY = 0., tF4hfY = 0., // for load vector (heat flux components for y-direction)
           tK11convX = 0., tK21convX = 0., tK31convX = 0., tK41convX = 0., tK22convX = 0., // for symetric matrix (convection components for x-direction)
           tK32convX = 0., tK42convX = 0., tK33convX = 0., tK43convX = 0., tK44convX = 0., // for symetric matrix (convection components for x-direction)
           tK11convY = 0., tK21convY = 0., tK31convY = 0., tK41convY = 0., tK22convY = 0., // for symetric matrix (convection components for y-direction)
           tK32convY = 0., tK42convY = 0., tK33convY = 0., tK43convY = 0., tK44convY = 0., // for symetric matrix (convection components for y-direction)
           tF1convX = 0., tF2convX = 0., tF3convX = 0., tF4convX = 0., // for load vector (convection components for x-direction)
           tF1convY = 0., tF2convY = 0., tF3convY = 0., tF4convY = 0., // for load vector (convection components for y-direction)
           tF1radX = 0., tF2radX = 0., tF3radX = 0., tF4radX = 0., // for load vector (radiation components for x-direction)
           tF1radY = 0., tF2radY = 0., tF3radY = 0., tF4radY = 0.; // for load vector (radiation components for y-direction)

    // set zeros
    for(int i = 0; i < mAHeight; i++)
        for(int j = 0; j < mAWidth; j++)
            mpA[i][j]=0.;

    // set stiffness matrix and load vector
    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        // set nodes numbers for the current element
        tLoLeftNo = ttE->getNLoLeftPtr()->getNo();
        tLoRightNo = ttE->getNLoRightPtr()->getNo();
        tUpLeftNo = ttE->getNUpLeftPtr()->getNo();
        tUpRightNo = ttE->getNUpRightPtr()->getNo();

        // set element size
        tElemWidth = fabs(ttE->getNLoLeftPtr()->getX() - ttE->getNLoRightPtr()->getX());
        tElemHeight = fabs(ttE->getNLoLeftPtr()->getY() - ttE->getNUpLeftPtr()->getY());

        // set assistant values
        std::vector<Box2D> tVecBox = (this->geometry)->getLeafsBoundingBoxes(); // geometry->extract(GeometryObject::PredicateHasClass("active"));
        Vec<2, double> tSize;
        for (Box2D tBox: tVecBox)
        {
            if (tBox.includes(vec(ttE->getX(), ttE->getY())))
            {
                tSize = tBox.size();
                break;
            }
        }
        tKXAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first;
        tKYAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second;

        // load vector: heat densities
        tF = 0.25 * tElemWidth * tElemHeight * 1e-12 * mHeatDensities[ttE->getNo()-1]; // 1e-12 -> to transform um*um into m*m

        // boundary condition: heat flux
        if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNLoRightPtr()->ifHFConst() ) // heat flux on bottom edge of the element
        {
            tF1hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getHF(); // 1e-6 -> to transform um into m
            tF2hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getHF();
        }
        if ( ttE->getNUpLeftPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on top edge of the element
        {
            tF3hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getHF();
            tF4hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getHF();
        }
        if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNUpLeftPtr()->ifHFConst() ) // heat flux on left edge of the element
        {
            tF1hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getHF();
            tF4hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getHF();
        }
        if ( ttE->getNLoRightPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on right edge of the element
        {
            tF2hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getHF();
            tF3hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getHF();
        }

        // boundary condition: convection
        if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNLoRightPtr()->ifConvection() ) // convection on bottom edge of the element
        {
            tF1convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1(); // 1e-6 -> to transform um into m
            tF2convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
            tK22convX = tK11convX = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNLoRightPtr()->getConvCoeff()) * tElemWidth / 3.;
            tK21convX = 0.5 * tK22convX;
        }
        if ( ttE->getNUpLeftPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on top edge of the element
        {
            tF3convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
            tF4convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
            tK44convX = tK33convX = (ttE->getNUpLeftPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemWidth / 3.;
            tK43convX = 0.5 * tK44convX;
        }
        if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNUpLeftPtr()->ifConvection() ) // convection on left edge of the element
        {
            tF1convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1();
            tF4convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
            tK44convY = tK11convY = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNUpLeftPtr()->getConvCoeff()) * tElemHeight / 3.;
            tK41convY = 0.5 * tK44convY;
        }
        if ( ttE->getNLoRightPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on right edge of the element
        {
            tF2convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
            tF3convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
            tK33convY = tK22convY = (ttE->getNLoRightPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemHeight / 3.;
            tK32convY = 0.5 * tK33convY;
        }

        // boundary condition: radiation
        if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNLoRightPtr()->ifRadiation() ) // radiation on bottom edge of the element
        {
            tF1radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4)); // 1e-6 -> to transform um into m
            tF2radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
        }
        if ( ttE->getNUpLeftPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on top edge of the element
        {
            tF3radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
            tF4radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
        }
        if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNUpLeftPtr()->ifRadiation() ) // radiation on left edge of the element
        {
            tF1radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4));
            tF4radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
        }
        if ( ttE->getNLoRightPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on right edge of the element
        {
            tF2radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
            tF3radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
        }

        // set symetric matrix components
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;

        // set stiffness matrix
        mpA[tLoLeftNo-1][mAWidth-2] +=  (tK11 + tK11convX + tK11convY);
        mpA[tLoRightNo-1][mAWidth-2] +=  (tK22 + tK22convX + tK22convY);
        mpA[tUpRightNo-1][mAWidth-2] += (tK33 + tK33convX + tK33convY);
        mpA[tUpLeftNo-1][mAWidth-2] += (tK44 + tK44convX + tK44convY);

        tLoRightNo > tLoLeftNo ? (tFstArg = tLoRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tLoRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK21 + tK21convX + tK21convY);

        tUpRightNo > tLoLeftNo ? (tFstArg = tUpRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK31 + tK31convX + tK31convY);

        tUpLeftNo > tLoLeftNo ? (tFstArg = tUpLeftNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK41 + tK41convX + tK41convY);

        tUpRightNo > tLoRightNo ? (tFstArg = tUpRightNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK32 + tK32convX + tK32convY);

        tUpLeftNo > tLoRightNo ? (tFstArg = tUpLeftNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK42 + tK42convX + tK42convY);

        tUpLeftNo > tUpRightNo ? (tFstArg = tUpLeftNo, tSecArg = tUpRightNo) : (tFstArg = tUpRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (tK43 + tK43convX + tK43convY);

        // set load vector
        mpA[tLoLeftNo-1][mAWidth-1]  += (tF + tF1hfX + tF1hfY + tF1convX + tF1convY + tF1radX + tF1radY);
        mpA[tLoRightNo-1][mAWidth-1] += (tF + tF2hfX + tF2hfY + tF2convX + tF2convY + tF2radX + tF2radY);
        mpA[tUpRightNo-1][mAWidth-1] += (tF + tF3hfX + tF3hfY + tF3convX + tF3convY + tF3radX + tF3radY);
        mpA[tUpLeftNo-1][mAWidth-1]  += (tF + tF4hfX + tF4hfY + tF4convX + tF4convY + tF4radX + tF4radY);
    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getT()*mBigNum;
        }
}

template<> void FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>::setMatrix()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting matrix...");

    std::vector<Element2D>::const_iterator ttE = mElements.begin();
    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    size_t tLoLeftNo, tLoRightNo, tUpLeftNo, tUpRightNo, // nodes numbers in current element
           tFstArg, tSecArg; // assistant values

    double tKXAssist, tKYAssist, tElemWidth, tElemHeight, tF, // assistant values to set stiffness matrix
           tK11, tK21, tK31, tK41, tK22, tK32, tK42, tK33, tK43, tK44, // local symetric matrix components
           tF1hfX = 0., tF2hfX = 0., tF3hfX = 0., tF4hfX = 0., // for load vector (heat flux components for x-direction)
           tF1hfY = 0., tF2hfY = 0., tF3hfY = 0., tF4hfY = 0., // for load vector (heat flux components for y-direction)
           tK11convX = 0., tK21convX = 0., tK31convX = 0., tK41convX = 0., tK22convX = 0., // for symetric matrix (convection components for x-direction)
           tK32convX = 0., tK42convX = 0., tK33convX = 0., tK43convX = 0., tK44convX = 0., // for symetric matrix (convection components for x-direction)
           tK11convY = 0., tK21convY = 0., tK31convY = 0., tK41convY = 0., tK22convY = 0., // for symetric matrix (convection components for y-direction)
           tK32convY = 0., tK42convY = 0., tK33convY = 0., tK43convY = 0., tK44convY = 0., // for symetric matrix (convection components for y-direction)
           tF1convX = 0., tF2convX = 0., tF3convX = 0., tF4convX = 0., // for load vector (convection components for x-direction)
           tF1convY = 0., tF2convY = 0., tF3convY = 0., tF4convY = 0., // for load vector (convection components for y-direction)
           tF1radX = 0., tF2radX = 0., tF3radX = 0., tF4radX = 0., // for load vector (radiation components for x-direction)
           tF1radY = 0., tF2radY = 0., tF3radY = 0., tF4radY = 0.; // for load vector (radiation components for y-direction)

    // set zeros
    for(int i = 0; i < mAHeight; i++)
        for(int j = 0; j < mAWidth; j++)
            mpA[i][j]=0.;

    // set stiffness matrix and load vector
    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        // set nodes numbers for the current element
        tLoLeftNo = ttE->getNLoLeftPtr()->getNo();
        tLoRightNo = ttE->getNLoRightPtr()->getNo();
        tUpLeftNo = ttE->getNUpLeftPtr()->getNo();
        tUpRightNo = ttE->getNUpRightPtr()->getNo();

        // set element size
        tElemWidth = fabs(ttE->getNLoLeftPtr()->getX() - ttE->getNLoRightPtr()->getX());
        tElemHeight = fabs(ttE->getNLoLeftPtr()->getY() - ttE->getNUpLeftPtr()->getY());

        // set assistant values
        std::vector<Box2D> tVecBox = (this->geometry)->getLeafsBoundingBoxes(); // geometry->extract(GeometryObject::PredicateHasClass("active"));
        Vec<2, double> tSize;
        for (Box2D tBox: tVecBox)
        {
            if (tBox.includes(vec(ttE->getX(), ttE->getY())))
            {
                tSize = tBox.size();
                break;
            }
        }
        tKXAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first;
        tKYAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second;

        // set load vector: heat densities
        tF = 0.25 * tElemWidth * tElemHeight * 1e-12 * mHeatDensities[ttE->getNo()-1]; // 1e-12 -> to transform um*um into m*m

        // boundary condition: heat flux
        if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNLoRightPtr()->ifHFConst() ) // heat flux on bottom edge of the element
        {
            tF1hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getHF(); // 1e-6 -> to transform um into m
            tF2hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getHF();
        }
        if ( ttE->getNUpLeftPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on top edge of the element
        {
            tF3hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getHF();
            tF4hfX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getHF();
        }
        if ( ttE->getNLoLeftPtr()->ifHFConst() && ttE->getNUpLeftPtr()->ifHFConst() ) // heat flux on left edge of the element
        {
            tF1hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getHF();
            tF4hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getHF();
        }
        if ( ttE->getNLoRightPtr()->ifHFConst() && ttE->getNUpRightPtr()->ifHFConst() ) // heat flux on right edge of the element
        {
            tF2hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getHF();
            tF3hfY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getHF();
        }

        // boundary condition: convection
        if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNLoRightPtr()->ifConvection() ) // convection on bottom edge of the element
        {
            tF1convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1(); // 1e-6 -> to transform um into m
            tF2convX = 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
            tK22convX = tK11convX = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNLoRightPtr()->getConvCoeff()) * tElemWidth / 3.;
            tK21convX = 0.5 * tK22convX;
        }
        if ( ttE->getNUpLeftPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on top edge of the element
        {
            tF3convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
            tF4convX = 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
            tK44convX = tK33convX = (ttE->getNUpLeftPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemWidth / 3.;
            tK43convX = 0.5 * tK44convX;
        }
        if ( ttE->getNLoLeftPtr()->ifConvection() && ttE->getNUpLeftPtr()->ifConvection() ) // convection on left edge of the element
        {
            tF1convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getConvCoeff() * ttE->getNLoLeftPtr()->getTAmb1();
            tF4convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getConvCoeff() * ttE->getNUpLeftPtr()->getTAmb1();
            tK44convY = tK11convY = (ttE->getNLoLeftPtr()->getConvCoeff() + ttE->getNUpLeftPtr()->getConvCoeff()) * tElemHeight / 3.;
            tK41convY = 0.5 * tK44convY;
        }
        if ( ttE->getNLoRightPtr()->ifConvection() && ttE->getNUpRightPtr()->ifConvection() ) // convection on right edge of the element
        {
            tF2convY = 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getConvCoeff() * ttE->getNLoRightPtr()->getTAmb1();
            tF3convY = 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getConvCoeff() * ttE->getNUpRightPtr()->getTAmb1();
            tK33convY = tK22convY = (ttE->getNLoRightPtr()->getConvCoeff() + ttE->getNUpRightPtr()->getConvCoeff()) * tElemHeight / 3.;
            tK32convY = 0.5 * tK33convY;
        }

        // boundary condition: radiation
        if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNLoRightPtr()->ifRadiation() ) // radiation on bottom edge of the element
        {
            tF1radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4)); // 1e-6 -> to transform um into m
            tF2radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
        }
        if ( ttE->getNUpLeftPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on top edge of the element
        {
            tF3radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
            tF4radX = - 0.5 * tElemWidth * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
        }
        if ( ttE->getNLoLeftPtr()->ifRadiation() && ttE->getNUpLeftPtr()->ifRadiation() ) // radiation on left edge of the element
        {
            tF1radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoLeftPtr()->getT(),4) - pow(ttE->getNLoLeftPtr()->getTAmb2(),4));
            tF4radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpLeftPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpLeftPtr()->getT(),4) - pow(ttE->getNUpLeftPtr()->getTAmb2(),4));
        }
        if ( ttE->getNLoRightPtr()->ifRadiation() && ttE->getNUpRightPtr()->ifRadiation() ) // radiation on right edge of the element
        {
            tF2radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNLoRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNLoRightPtr()->getT(),4) - pow(ttE->getNLoRightPtr()->getTAmb2(),4));
            tF3radY = - 0.5 * tElemHeight * 1e-6 * ttE->getNUpRightPtr()->getEmissivity() * cPhys::SB * (pow(ttE->getNUpRightPtr()->getT(),4) - pow(ttE->getNUpRightPtr()->getTAmb2(),4));
        }

        // set symetric matrix components
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;

        // set stiffness matrix
        mpA[tLoLeftNo-1][mAWidth-2] += ( ttE->getX() * (tK11 + tK11convX + tK11convY) );
        mpA[tLoRightNo-1][mAWidth-2] += ( ttE->getX() * (tK22 + tK22convX + tK22convY) );
        mpA[tUpRightNo-1][mAWidth-2] += ( ttE->getX() * (tK33 + tK33convX + tK33convY) );
        mpA[tUpLeftNo-1][mAWidth-2] += ( ttE->getX() * (tK44 + tK44convX + tK44convY) );

        tLoRightNo > tLoLeftNo ? (tFstArg = tLoRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tLoRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK21 + tK21convX + tK21convY) );

        tUpRightNo > tLoLeftNo ? (tFstArg = tUpRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK31 + tK31convX + tK31convY) );

        tUpLeftNo > tLoLeftNo ? (tFstArg = tUpLeftNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK41 + tK41convX + tK41convY) );

        tUpRightNo > tLoRightNo ? (tFstArg = tUpRightNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK32 + tK32convX + tK32convY) );

        tUpLeftNo > tLoRightNo ? (tFstArg = tUpLeftNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK42 + tK42convX + tK42convY) );

        tUpLeftNo > tUpRightNo ? (tFstArg = tUpLeftNo, tSecArg = tUpRightNo) : (tFstArg = tUpRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += ( ttE->getX() * (tK43 + tK43convX + tK43convY) );

        // set load vector
        mpA[tLoLeftNo-1][mAWidth-1]  += ( ttE->getX() * (tF + tF1hfX + tF1hfY + tF1convX + tF1convY + tF1radX + tF1radY) );
        mpA[tLoRightNo-1][mAWidth-1] += ( ttE->getX() * (tF + tF2hfX + tF2hfY + tF2convX + tF2convY + tF2radX + tF2radY) );
        mpA[tUpRightNo-1][mAWidth-1] += ( ttE->getX() * (tF + tF3hfX + tF3hfY + tF3convX + tF3convY + tF3radX + tF3radY) );
        mpA[tUpLeftNo-1][mAWidth-1]  += ( ttE->getX() * (tF + tF4hfX + tF4hfY + tF4convX + tF4convY + tF4radX + tF4radY) );
    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getT()*mBigNum;
        }
}

template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::runCalc()
{
    //if (!isInitialized()) std::cout << "First calc.\n";
    ///else "Cont. calc.\n";
    this->initCalculation();
    //if (!inHeats.changed) return;

    if (mLogs)
        writelog(LOG_INFO, "Starting thermal calculations...");

    setSolver();

    setHeatDensities();

    if (mLogs)
        writelog(LOG_INFO, "Running solver...");

    std::vector<double>::const_iterator ttTCorr;
    int tLoop = 1, tInfo = 1;
    double tTCorr = mTBigCorr; // much bigger than calculated corrections

    while ( (tLoop <= mLoopLim) && (tTCorr > mTCorrLim) )
    {
        setMatrix();

        tInfo = solveMatrix(mpA, mNodes.size(), (this->mesh)->minorAxis().size()+2);
        if (!tInfo)
        {
            // update
            updNodes();
            updElements();

            // find max correction
            std::vector<double> tB(mAHeight, 0.);

            for (int i = 0; i < mAHeight; i++)
                tB.at(i) = mpA[i][mAWidth-1]; // mpA[i][mAWidth-1] - here are new values of temperature

            ttTCorr = std::max_element(mTCorr.begin(), mTCorr.end());

            tTCorr = *ttTCorr;

            mLoopNo++;

            // show max correction
            writelog(LOG_DETAIL, "Loop no: %1%(%2%), max. corr. for T: %3%", tLoop, mLoopNo, tTCorr);

            tLoop++;
        }
        else if (tInfo < 0)
            writelog(LOG_ERROR, "Wrong value of new temperature");
        else
            writelog(LOG_ERROR, "Wrong solver matrix");
    }

    if (mLogs)
        showNodes();

    double tTCorrOut = saveTemperatures();

    saveHeatFluxes();

    if (mLogs)
        showTemperatures();

    if (mLogs)
        showHeatFluxes();

    delSolver();

    if (mLogs)
        writelog(LOG_INFO, "Temperature calculations completed");

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();

    return tTCorrOut;
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd()) {
        std::string param = source.getNodeName();

        if (param == "Tconst")
            this->readBoundaryConditions(manager, source, mTConst);
        else if (param == "HFconst")
            this->readBoundaryConditions(manager, source, mHFConst);
        else if (param == "convection")
            this->readBoundaryConditions(manager, source, mConvection);
        else if (param == "radiation")
            this->readBoundaryConditions(manager, source, mRadiation);
        else if (param == "Tinit")
        {
            mTInit = source.requireAttribute<double>("value");
            source.requireTagEnd();
        }
        else if (param == "looplim")
        {
            mLoopLim = source.requireAttribute<int>("value");
            source.requireTagEnd();
        }
        else if (param == "Tcorrlim")
        {
            mTCorrLim = source.requireAttribute<double>("value");
            source.requireTagEnd();
        }
        else if (param == "Tbigcorr")
        {
            mTBigCorr = source.requireAttribute<double>("value");
            source.requireTagEnd();
        }
        else if (param == "bignum")
        {
            mBigNum = source.requireAttribute<double>("value");
            source.requireTagEnd();
        }
        else if (param == "logs")
        {
            mLogs = source.requireAttribute<bool>("value");
            source.requireTagEnd();
        }
        else
            this->parseStandardConfiguration(source, manager);
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::updNodes()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating nodes...");

    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        mTCorr[ttN->getNo()-1] = fabs( ttN->getT() - mpA[ttN->getNo()-1][mAWidth-1] ); // calculate corrections
        if (!ttN->ifTConst())
            ttN->setT( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of temperature
        ttN++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::updElements()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating elements...");

    for (std::vector<Element2D>::iterator ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        ttE->setT();
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::showNodes()
{
    writelog(LOG_INFO, "Showing nodes...");

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", x: " << ttN->getX() << ", y: " << ttN->getY() << ", T: " << ttN->getT() << std::endl; // TEST
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::showElements()
{
    writelog(LOG_INFO, "Showing elements...");

    std::vector<Element2D>::const_iterator ttE;

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        std::cout << "Element no: " << ttE->getNo()
                     << ", BL: " << ttE->getNLoLeftPtr()->getNo() << " (" << ttE->getNLoLeftPtr()->getX() << "," << ttE->getNLoLeftPtr()->getY() << ")"
                     << ", BR: " << ttE->getNLoRightPtr()->getNo() << " (" << ttE->getNLoRightPtr()->getX() << "," << ttE->getNLoRightPtr()->getY() << ")"
                     << ", TL: " << ttE->getNUpLeftPtr()->getNo() << " (" << ttE->getNUpLeftPtr()->getX() << "," << ttE->getNUpLeftPtr()->getY() << ")"
                     << ", TR: " << ttE->getNUpRightPtr()->getNo() << " (" << ttE->getNUpRightPtr()->getX() << "," << ttE->getNUpRightPtr()->getY() << ")"
                     << std::endl; // TEST
}

template<typename Geometry2Dtype> double FiniteElementMethodThermal2DSolver<Geometry2Dtype>::saveTemperatures()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving temperatures...");

    std::vector<Node2D>::const_iterator ttN;

    // mTemperatures.reset(mNodes.size());

    double tCorr = 0.;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
    {
        double tC = fabs(mTemperatures[ttN->getNo()-1] - ttN->getT());
        if (tC > tCorr)
            tCorr = tC;
        mTemperatures[ttN->getNo()-1] = ttN->getT();
    }

    return tCorr;
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::saveHeatFluxes()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving heat fluxes...");

    std::vector<Element2D>::const_iterator ttE;

    mHeatFluxes.reset(mElements.size());

    std::vector<Box2D> tVecBox = (this->geometry)->getLeafsBoundingBoxes();

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        Vec<2, double> tSize;
        for (Box2D tBox: tVecBox)
        {
            if (tBox.includes(vec(ttE->getX(), ttE->getY())))
            {
                tSize = tBox.size();
                break;
            }
        }
        mHeatFluxes[ttE->getNo()-1] = vec(
            - ((this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first) * ttE->getdTdX() * 1e6, // 1e6 - from um to m
            - ((this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second) * ttE->getdTdY() * 1e6 ); // 1e6 - from um to m
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::showTemperatures()
{
    std::cout << "Showing temperatures... " << mTemperatures<< std::endl;
}

template<typename Geometry2Dtype> void FiniteElementMethodThermal2DSolver<Geometry2Dtype>::showHeatFluxes()
{
    std::cout << "Showing heat fluxes... " << mHeatFluxes<< std::endl;
}

template<typename Geometry2Dtype> int FiniteElementMethodThermal2DSolver<Geometry2Dtype>::solveMatrix(double **ipA, long iN, long iBandWidth)
{
    if (mLogs)
        writelog(LOG_INFO, "Solving matrix...");

    long m, k, i, j, poc, kon, q, mn;
    double SUM;

    // creating L-matrix
    for( j = 0; j < iN; j++)
    {
        poc = j - (iBandWidth-1);
        if( poc < 0 ) poc = 0;
        kon = j + (iBandWidth-1);
        if( kon > (iN-1) ) kon = iN-1;
        SUM = 0.0;
        for( k = poc; k <= (j-1); k++)
        {
            m = k - j + (iBandWidth-1);
            SUM += ipA[j][m]*ipA[j][m];
        }
        m = j - j + (iBandWidth-1);
        ipA[j][m] = sqrt(ipA[j][m] - SUM);
        for( i = j+1; i <= kon; i++)
        {
            if( i > (iBandWidth-1)) mn = 1;
            else					mn = 0;
            poc += 1*mn;
            SUM = 0.0;
            for( k = poc; k <= (j-1); k++)
            {
                m = k - i + (iBandWidth-1);
                q = k - j + (iBandWidth-1);
                SUM += ipA[i][m]*ipA[j][q];
            }
            m = j - i + (iBandWidth-1);
            q = j - j + (iBandWidth-1);
            ipA[i][m] = ( ipA[i][m] - SUM ) / ipA[j][q];
        }
    }

    // solving LY = B
    for( j = 0; j < iN; j++)
    {
        poc = j - (iBandWidth-1);
        if( poc < 0 ) poc = 0;
        SUM = 0.0;
        for( k = poc; k <= (j-1); k++)
        {
            m = k - j + (iBandWidth-1);
            SUM += ipA[j][m]*ipA[k][iBandWidth];
        }
        m = j - j + (iBandWidth-1);
        ipA[j][iBandWidth] = ( ipA[j][iBandWidth] - SUM ) / ipA[j][m];
    }

    // solving L^TX=Y
    for( j = (iN-1); j >= 0; j--)
    {
        kon = j + (iBandWidth-1);
        if( kon > (iN-1) ) kon = iN-1;
        SUM = 0.0;
        for( k = j+1; k <= kon; k++)
        {
            m = j - k + (iBandWidth-1);
            SUM += ipA[k][m]*ipA[k][iBandWidth];
        }
        m = j - j + (iBandWidth-1);
        ipA[j][iBandWidth] = ( ipA[j][iBandWidth] - SUM ) / ipA[j][m];
    }

    return 0;
}

template<typename Geometry2Dtype> DataVector<const double> FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getTemperatures(const MeshD<2> &dst_mesh, InterpolationMethod method) const {
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mTemperatures, dst_mesh, method);
}

template<typename Geometry2Dtype> DataVector<const Vec<2> > FiniteElementMethodThermal2DSolver<Geometry2Dtype>::getHeatFluxes(const MeshD<2> &dst_mesh, InterpolationMethod method) const {
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mHeatFluxes, dst_mesh, method);
}

template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCartesian>::getClassName() const { return "CartesianFEM"; }

template<> std::string FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>::getClassName() const { return "CylindricalFEM"; }

template struct FiniteElementMethodThermal2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodThermal2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::thermal
