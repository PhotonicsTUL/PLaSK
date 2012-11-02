#include <limits>

#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2Dtype> FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::FiniteElementMethodElectrical2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2Dtype, RectilinearMesh2D>(name),
    mpA(nullptr),
    mVChange("absolute"),
    mLoopLim(5),
    mVCorrLim(0.01),
    mBigNum(1e15),
    mJs(1.),
    mBeta(20.),
    mCondJuncX0(1e-6),
    mCondJuncY0(5.),
    mCondPcontact(5.),
    mCondNcontact(50.),
    mLogs(false),
    mLoopNo(0),
    outPotential(this, &FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getCurrentDensities),
    outHeatDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getHeatDensities)
{
    mNodes.clear();
    mElements.clear();
    mPotentials.reset();
    mCurrentDensities.reset();
    mTemperatures.reset();
    mHeatDensities.reset();

    inTemperature = 300.;
}

template<typename Geometry2Dtype> FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::~FiniteElementMethodElectrical2DSolver()
{
    if (mpA)
    {
        for (int i = 0; i < mAHeight; i++)
            delete [] mpA[i];
        delete [] mpA;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::onInitialize() // In this function check if geometry and mesh are set
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    this->setNodes();
    this->setSolver();
    this->setElements();
    mPotentials.reset(mNodes.size(), 0.);
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    mNodes.clear();
    mElements.clear();
    mPotentials.reset();
    mCurrentDensities.reset();
    mTemperatures.reset();
    mHeatDensities.reset();
    // Make sure that no provider returns any value.
    // If this method has been called, before next computations, onInitialize will be called.
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setLoopLim(int iLoopLim) { mLoopLim = iLoopLim; }
template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setVCorrLim(double iVCorrLim) { mVCorrLim = iVCorrLim; }
template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setBigNum(double iBigNum) { mBigNum = iBigNum; }

template<typename Geometry2Dtype> int FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getLoopLim() { return mLoopLim; }
template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getVCorrLim() { return mVCorrLim; }
template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getBigNum() { return mBigNum; }

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setNodes()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Setting nodes...");

    size_t tNo = 1; // node number

    Node2D* tpN = NULL;

    auto tVConst = mVConst.get(this->mesh);

    for(plask::RectilinearMesh2D::iterator vec_it = (this->mesh)->begin(); vec_it != (this->mesh)->end(); ++vec_it) // loop through all nodes given in the correct iteration order
    {
        double x = vec_it->ee_x();
        double y = vec_it->ee_y();

        std::size_t i = vec_it.getIndex();
        auto it = tVConst.find(i);
        if (it != tVConst.end())
            tpN = new Node2D(tNo, x, y, it->value, true);
        else
            tpN = new Node2D(tNo, x, y, 0., false);

        mNodes.push_back(*tpN);

        delete tpN;
        tNo++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setElements()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Setting elements...");

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
            //tpE->setT();
            mElements.push_back(*tpE);

            delete tpE;
            tNo++;
            ttN++;
        }
        ttN++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setTemperatures()
{
    auto iMesh = (this->mesh)->getMidpointsMesh();
    mTemperatures = inTemperature(iMesh);
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::setSolver()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Setting solver...");

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

    //std::cout << "Main matrix width: " << mAWidth << std::endl; // TEST
    //std::cout << "Main matrix height: " << mAHeight << std::endl; // TEST
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::delSolver()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Deleting solver...");

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;
}

template<> void FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::setMatrix()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Setting matrix...");

    std::vector<Element2D>::iterator ttE = mElements.begin();
    std::vector<Node2D>::iterator ttN = mNodes.begin();

    size_t tLoLeftNo, tLoRightNo, tUpLeftNo, tUpRightNo, // nodes numbers in current element
           tFstArg, tSecArg; // assistant values

    double tKXAssist, tKYAssist, tElemWidth, tElemHeight, /*tF,*/ // assistant values to set stiffness matrix
           tK11, tK21, tK31, tK41, tK22, tK32, tK42, tK33, tK43, tK44; // local symetric matrix components

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
        if ((this->geometry)->hasRoleAt("active", vec(ttE->getX(), ttE->getY()))) // TODO
        {
            tKXAssist = mCondJuncX0;
            if (!mLoopNo)
                ttE->setCondJuncY(mCondJuncY0);
            else
            {
                // TODO: no good enough
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

                double tJy = ttE->getCondJuncY() * fabs(ttE->getdVdY()) * 1e6; // 1e6 - from um to m
                double tDact = tSize.ee_y() * 1e-6; // 1e-6 - from um to m
                ttE->setCondJuncY( (mBeta * tJy * tDact) / log(tJy / mJs + 1.) );
            }
            tKYAssist = ttE->getCondJuncY();
        }
        else if ((this->geometry)->hasRoleAt("p-contact", vec(ttE->getX(), ttE->getY())))
            tKYAssist = tKXAssist = mCondPcontact;
        else if ((this->geometry)->hasRoleAt("n-contact", vec(ttE->getX(), ttE->getY())))
            tKYAssist = tKXAssist = mCondNcontact;
        else
        {
            tKXAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).first;
            tKYAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).second;
        }

        // set load vector
        //tF = 0.; //0.25 * tElemWidth * tElemHeight * 1e-12 * 0.; // 1e-12 -> to transform um*um into m*m

        // set symetric matrix components
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;

        // set stiffness matrix
        mpA[tLoLeftNo-1][mAWidth-2] +=  tK11;
        mpA[tLoRightNo-1][mAWidth-2] +=  tK22;
        mpA[tUpRightNo-1][mAWidth-2] += tK33;
        mpA[tUpLeftNo-1][mAWidth-2] += tK44;

        tLoRightNo > tLoLeftNo ? (tFstArg = tLoRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tLoRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK21;

        tUpRightNo > tLoLeftNo ? (tFstArg = tUpRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK31;

        tUpLeftNo > tLoLeftNo ? (tFstArg = tUpLeftNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK41;

        tUpRightNo > tLoRightNo ? (tFstArg = tUpRightNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK32;

        tUpLeftNo > tLoRightNo ? (tFstArg = tUpLeftNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK42;

        tUpLeftNo > tUpRightNo ? (tFstArg = tUpLeftNo, tSecArg = tUpRightNo) : (tFstArg = tUpRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += tK43;

        // set load vector
        /*mpA[tLoLeftNo-1][mAWidth-1]  += tF;
        mpA[tLoRightNo-1][mAWidth-1] += tF;
        mpA[tUpRightNo-1][mAWidth-1] += tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += tF;*/

    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifVConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getV()*mBigNum;
        }
}

template<> void FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::setMatrix()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Setting matrix...");

    std::vector<Element2D>::iterator ttE = mElements.begin();
    std::vector<Node2D>::iterator ttN = mNodes.begin();

    size_t tLoLeftNo, tLoRightNo, tUpLeftNo, tUpRightNo, // nodes numbers in current element
           tFstArg, tSecArg; // assistant values

    double tKXAssist, tKYAssist, tElemWidth, tElemHeight, /*tF,*/ // assistant values to set stiffness matrix
           tK11, tK21, tK31, tK41, tK22, tK32, tK42, tK33, tK43, tK44; // local symetric matrix components

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
        if ((this->geometry)->hasRoleAt("active", vec(ttE->getX(), ttE->getY()))) // TODO
        {
            tKXAssist = mCondJuncX0;
            if (!mLoopNo)
                ttE->setCondJuncY(mCondJuncY0);
            else
            {
                // TODO: no good enough
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

                double tJy = ttE->getCondJuncY() * fabs(ttE->getdVdY()) * 1e6; // 1e6 - from um to m
                double tDact = tSize.ee_y() * 1e-6; // 1e-6 - from um to m
                ttE->setCondJuncY( (mBeta * tJy * tDact) / log(tJy / mJs + 1.) );
            }
            tKYAssist = ttE->getCondJuncY();
        }
        else if ((this->geometry)->hasRoleAt("p-contact", vec(ttE->getX(), ttE->getY())))
            tKYAssist = tKXAssist = mCondPcontact;
        else if ((this->geometry)->hasRoleAt("n-contact", vec(ttE->getX(), ttE->getY())))
            tKYAssist = tKXAssist = mCondNcontact;
        else
        {
            tKXAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).first;
            tKYAssist = (this->geometry)->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).second;
        }

        // set load vector
        //tF = 0.; //0.25 * tElemWidth * tElemHeight * 1e-12 * 0.; // 1e-12 -> to transform um*um into m*m

        // set symetric matrix components
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;

        // set stiffness matrix
        mpA[tLoLeftNo-1][mAWidth-2] +=  (ttE->getX() * tK11);
        mpA[tLoRightNo-1][mAWidth-2] +=  (ttE->getX() * tK22);
        mpA[tUpRightNo-1][mAWidth-2] += (ttE->getX() * tK33);
        mpA[tUpLeftNo-1][mAWidth-2] += (ttE->getX() * tK44);

        tLoRightNo > tLoLeftNo ? (tFstArg = tLoRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tLoRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK21);

        tUpRightNo > tLoLeftNo ? (tFstArg = tUpRightNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK31);

        tUpLeftNo > tLoLeftNo ? (tFstArg = tUpLeftNo, tSecArg = tLoLeftNo) : (tFstArg = tLoLeftNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK41);

        tUpRightNo > tLoRightNo ? (tFstArg = tUpRightNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpRightNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK32);

        tUpLeftNo > tLoRightNo ? (tFstArg = tUpLeftNo, tSecArg = tLoRightNo) : (tFstArg = tLoRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK42);

        tUpLeftNo > tUpRightNo ? (tFstArg = tUpLeftNo, tSecArg = tUpRightNo) : (tFstArg = tUpRightNo, tSecArg = tUpLeftNo);
        mpA[tFstArg-1][mAWidth-2-(tFstArg-tSecArg)] += (ttE->getX() * tK43);

        // set load vector
        /*mpA[tLoLeftNo-1][mAWidth-1]  += tF;
        mpA[tLoRightNo-1][mAWidth-1] += tF;
        mpA[tUpRightNo-1][mAWidth-1] += tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += tF;*/

    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifVConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getV()*mBigNum;
        }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::doSomeSteps()
{
    this->initCalculation();

    if (mLogs)
        writelog(LOG_INFO, "Starting electrical calculations...");

    //setSolver();

    setTemperatures();

    if (mLogs)
        writelog(LOG_DETAIL, "Running solver...");
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::doMoreSteps()
{
    if (mLogs)
        showNodes();

    savePotentials();

    saveCurrentDensities();

    saveHeatDensities();

    if (mLogs)
        showPotentials();

    if (mLogs)
        showCurrentDensities();

    if (mLogs)
        showHeatDensities();

    //delSolver();

    if (mLogs)
        writelog(LOG_INFO, "Potential calculations completed");

    outPotential.fireChanged();
    outCurrentDensity.fireChanged();
    outHeatDensity.fireChanged();
}

template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::runCalc(int iLoopLim)
{
    if (iLoopLim == 1) return runSingleCalc();
    mLoopLim = iLoopLim? iLoopLim : std::numeric_limits<int>::max();

    doSomeSteps();

    int tInfo = 1;
    std::vector<double>::const_iterator ttVCorr;
    int tLoop = 1;
    double tVCorr = 2.*mVCorrLim; // bigger than calculated corrections
    while ( (tLoop <= mLoopLim) && (tVCorr > mVCorrLim) )
    {
        setMatrix();

        tInfo = solveMatrix(mpA, mNodes.size(), (this->mesh)->minorAxis().size()+2);
        if (!tInfo)
        {
            // update
            updNodes();
            //updElements();

            tVCorr = mMaxAbsVCorr;

            mLoopNo++;

            // show max correction
            writelog(LOG_DATA, "Loop no: %d(%d), max. V upd: %8.6f (abs), %8.6f (rel)", tLoop, mLoopNo, mMaxAbsVCorr, mMaxRelVCorr/*tVCorr*/);

            tLoop++;
        }
        else if (tInfo < 0)
            writelog(LOG_ERROR, "Wrong value of new potential");
        else
            writelog(LOG_ERROR, "Wrong solver matrix");
    }

    doMoreSteps();

    return mMaxVCorr;
}

template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::runSingleCalc()
{
    doSomeSteps();

    int tInfo = 1;
    setMatrix();
    tInfo = solveMatrix(mpA, mNodes.size(), (this->mesh)->minorAxis().size()+2);
    if (!tInfo)
    {
        // update
        updNodes();
        //updElements();

        mLoopNo++;

        // show max correction
        writelog(LOG_DATA, "Loop no: %d, max. V upd: %8.6f (abs), %8.6f (rel)", mLoopNo, mMaxAbsVCorr, mMaxRelVCorr);
    }
    else if (tInfo < 0)
        writelog(LOG_ERROR, "Wrong value of new potential");
    else
        writelog(LOG_ERROR, "Wrong solver matrix");

    doMoreSteps();

    if (mVChange == "relative")
        return mMaxRelVCorr;
    else //if (mVChange == "absolute")
        return mMaxAbsVCorr;
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::loadConfiguration(XMLReader &source, Manager &manager)
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
                if (tValue == "absolute" || tValue == "abs") mVChange = "absolute"; //mCorrType = CORRECTION_ABSOLUTE;
                else if (tValue == "relative" || tValue == "rel") mVChange = "realtive"; //mCorrType = CORRECTION_RELATIVE;
                else throw XMLBadAttrException(source, "corrtype", *tCorrType, + "\"abs[olute]\" or \"rel[ative]\"");
            }
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            mBigNum = source.getAttribute<double>("bignum", mBigNum);
//             auto tAlgo = source.getAttribute("algorithm");
//             if (tAlgo) {
//                 std::string tValue = *tAlgo; boost::algorithm::to_lower(tValue);
//                 if (tValue == "slow") mAlgorithm = ALGORITHM_SLOW;
//                 else if (tValue == "block") mAlgorithm = ALGORITHM_SLOW;
//                 //else if (tValue == "iterative") mAlgorithm = ALGORITHM_ITERATIVE;
//                 else throw XMLBadAttrException(source, "algorithm", *tAlgo, + "\"block\" or \"slow\"");
//             }
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
            source.requireTagEnd();
        }

        else if (param == "contacts") {
            mCondPcontact = source.getAttribute<double>("p_cond", mCondPcontact);
            mCondNcontact = source.getAttribute<double>("n_cond", mCondNcontact);

        }

        else
            this->parseStandardConfiguration(source, manager);
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::updNodes()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Updating nodes...");

    std::vector<Node2D>::iterator ttN = mNodes.begin();

    mMaxAbsVCorr = 0.;
    mMaxRelVCorr = 0.;

    while (ttN != mNodes.end())
    {
        double tMaxAbsVCorr = fabs( ttN->getV() - mpA[ttN->getNo()-1][mAWidth-1] );
        if ((tMaxAbsVCorr > mMaxAbsVCorr) && !ttN->ifVConst())
            mMaxAbsVCorr = tMaxAbsVCorr;
        if ((mpA[ttN->getNo()-1][mAWidth-1]) && !ttN->ifVConst())
        {
            double tMaxRelVCorr = fabs( ttN->getV() - mpA[ttN->getNo()-1][mAWidth-1] ) * 100. / mpA[ttN->getNo()-1][mAWidth-1];
            if (tMaxRelVCorr > mMaxRelVCorr)
                mMaxRelVCorr = tMaxRelVCorr;
        }

        if (!ttN->ifVConst())
            ttN->setV( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of potential
        ttN++;
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::updElements()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Updating elements...");
}

template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getMaxAbsVCorr()
{
    return mMaxAbsVCorr;
}

template<typename Geometry2Dtype> double FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getMaxRelVCorr()
{
    return mMaxRelVCorr;
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::showNodes()
{
#ifndef NDEBUG
    writelog(LOG_DETAIL, "Showing nodes...");

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        writelog(LOG_DEBUG, "Node no: %1%, x: %2%, y: %3%, V: %4%", ttN->getNo(), ttN->getX(), ttN->getY(), ttN->getV());
#endif
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::showElements()
{
#ifndef NDEBUG
    writelog(LOG_DEBUG, "Showing elements...");

    std::vector<Element2D>::const_iterator ttE;

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        std::cout << "Element no: " << ttE->getNo()
                     << ", BL: " << ttE->getNLoLeftPtr()->getNo() << " (" << ttE->getNLoLeftPtr()->getX() << "," << ttE->getNLoLeftPtr()->getY() << ")"
                     << ", BR: " << ttE->getNLoRightPtr()->getNo() << " (" << ttE->getNLoRightPtr()->getX() << "," << ttE->getNLoRightPtr()->getY() << ")"
                     << ", TL: " << ttE->getNUpLeftPtr()->getNo() << " (" << ttE->getNUpLeftPtr()->getX() << "," << ttE->getNUpLeftPtr()->getY() << ")"
                     << ", TR: " << ttE->getNUpRightPtr()->getNo() << " (" << ttE->getNUpRightPtr()->getX() << "," << ttE->getNUpRightPtr()->getY() << ")"
                     << std::endl; // TEST
#endif
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::savePotentials()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Saving potentials...");

    std::vector<Node2D>::const_iterator ttN;

    mMaxVCorr = 0.;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
    {
        double tMaxVCorr = fabs(mPotentials[ttN->getNo()-1] - ttN->getV());
        if (tMaxVCorr > mMaxVCorr)
            mMaxVCorr = tMaxVCorr;
        mPotentials[ttN->getNo()-1] = ttN->getV();
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::saveCurrentDensities()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Saving current densities...");

    std::vector<Element2D>::const_iterator ttE;

    mCurrentDensities.reset(mElements.size());

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        if ((this->geometry)->hasRoleAt("insulator", vec(ttE->getX(), ttE->getY())))
            mCurrentDensities[ttE->getNo()-1] = vec(0., 0.);
        else if ((this->geometry)->hasRoleAt("active", vec(ttE->getX(), ttE->getY()))) // TODO
        {
            mCurrentDensities[ttE->getNo()-1] = vec(
                - mCondJuncX0 * ttE->getdVdX() * 0.1, // kA/cm^2
                - ttE->getCondJuncY() * ttE->getdVdY() * 0.1 ); // kA/cm^2
        }
        else if ((this->geometry)->hasRoleAt("p-contact", vec(ttE->getX(), ttE->getY())))
            mCurrentDensities[ttE->getNo()-1] = vec(
                - mCondPcontact * ttE->getdVdX() * 0.1, // kA/cm^2
                - mCondPcontact * ttE->getdVdY() * 0.1 ); // kA/cm^2
        else if ((this->geometry)->hasRoleAt("n-contact", vec(ttE->getX(), ttE->getY())))
            mCurrentDensities[ttE->getNo()-1] = vec(
                - mCondNcontact * ttE->getdVdX() * 0.1, // kA/cm^2
                - mCondNcontact * ttE->getdVdY() * 0.1 ); // kA/cm^2
        else
        {
            auto tCond = this->geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]);
            mCurrentDensities[ttE->getNo()-1] = vec(
                - tCond.first * ttE->getdVdX() * 0.1, // kA/cm^2
                - tCond.second * ttE->getdVdY() * 0.1 ); // kA/cm^2
        }
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::saveHeatDensities()
{
    if (mLogs)
        writelog(LOG_DETAIL, "Saving heat densities...");

    std::vector<Element2D>::const_iterator ttE;

    mHeatDensities.reset(mElements.size());

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        Vec<2> j = 1e7 * mCurrentDensities[ttE->getNo()-1];

        if ((this->geometry)->hasRoleAt("insulator", vec(ttE->getX(), ttE->getY())))
            mHeatDensities[ttE->getNo()-1] = 0.;
        else if ((this->geometry)->hasRoleAt("active", vec(ttE->getX(), ttE->getY()))) // TODO
        {
            std::vector<Box2D> tVecBox = (this->geometry)->getLeafsBoundingBoxes();
            Vec<2, double> tSize;
            for (Box2D tBox: tVecBox)
            {
                if (tBox.includes(vec(ttE->getX(), ttE->getY())))
                {
                    tSize = tBox.size();
                    break;
                }
            }

            mHeatDensities[ttE->getNo()-1] = phys::h_J * phys::c * fabs(j.ee_y()) / ( phys::qe * real(inWavelength())*1e-9 * 1e-6*tSize.ee_y() );
        }
        else if ((this->geometry)->hasRoleAt("p-contact", vec(ttE->getX(), ttE->getY())))
            mHeatDensities[ttE->getNo()-1] = (j.ee_x() * j.ee_x() + j.ee_y() * j.ee_y()) / mCondPcontact;
        else if ((this->geometry)->hasRoleAt("n-contact", vec(ttE->getX(), ttE->getY())))
            mHeatDensities[ttE->getNo()-1] = (j.ee_x() * j.ee_x() + j.ee_y() * j.ee_y()) / mCondNcontact;
        else
        {
            auto cond = this->geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]);
            mHeatDensities[ttE->getNo()-1] = j.ee_x() * j.ee_x() / cond.first + j.ee_y() * j.ee_y() / cond.second;
        }
    }
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::showPotentials()
{
    std::cout << "Showing potentials... " << mPotentials << std::endl;
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::showCurrentDensities()
{
    std::cout << "Showing current densities... " << mCurrentDensities << std::endl;
}

template<typename Geometry2Dtype> void FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::showHeatDensities()
{
    std::cout << "Showing heat densities... " << mHeatDensities << std::endl;
}

template<typename Geometry2Dtype> int FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::solveMatrix(double **ipA, long iN, long iBandWidth)
{
    if (mLogs)
        writelog(LOG_DETAIL, "Solving matrix...");

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

template<typename Geometry2Dtype> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getPotentials(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*(this->mesh), mPotentials, dst_mesh, method);
}

template<typename Geometry2Dtype> DataVector<const Vec<2> > FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getCurrentDensities(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mCurrentDensities, dst_mesh, method);
}

template<typename Geometry2Dtype> DataVector<const double> FiniteElementMethodElectrical2DSolver<Geometry2Dtype>::getHeatDensities(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*((this->mesh)->getMidpointsMesh()), mHeatDensities, dst_mesh, method);
}

template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.Fem2D"; }
template<> std::string FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.FemCyl"; }

template struct FiniteElementMethodElectrical2DSolver<Geometry2DCartesian>;
template struct FiniteElementMethodElectrical2DSolver<Geometry2DCylindrical>;

}}} // namespace plask::solvers::eletrical
