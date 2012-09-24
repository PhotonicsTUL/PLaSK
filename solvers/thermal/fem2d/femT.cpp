#include "femT.h"


namespace plask { namespace solvers { namespace thermal {

FiniteElementMethodThermalCartesian2DSolver::FiniteElementMethodThermalCartesian2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(5),
    mTCorrLim(0.1),
    mTBigCorr(1e5),
    mBigNum(1e15),
    mTAmb(300.),
    mLogs(false),
    mLoopNo(0),
    outTemperature(this, &FiniteElementMethodThermalCartesian2DSolver::getTemperatures),
    outHeatFlux(this, &FiniteElementMethodThermalCartesian2DSolver::getHeatFluxes)
{
    mNodes.clear();
    mElements.clear();
    mTemperatures.reset(0);
    mHeatFluxes.reset(0);
    mHeatDensities.reset();

    inHeatDensity = 0.;

}

FiniteElementMethodThermalCartesian2DSolver::~FiniteElementMethodThermalCartesian2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

void FiniteElementMethodThermalCartesian2DSolver::onInitialize() { // In this function check if geometry and mesh are set
    if (!geometry) throw NoGeometryException(getId());
    if (!mesh) throw NoMeshException(getId());
    setNodes();
    setElements();
}

void FiniteElementMethodThermalCartesian2DSolver::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
{
    mNodes.clear();
    mElements.clear();
    mTemperatures.reset(0);
    mHeatFluxes.reset(0);
    mHeatDensities.reset();
    // Make sure that no provider returns any value.
    // If this method has been called, before next computations, onInitialize will be called.
}

void FiniteElementMethodThermalCartesian2DSolver::setLoopLim(int iLoopLim) { mLoopLim = iLoopLim; }
void FiniteElementMethodThermalCartesian2DSolver::setTCorrLim(double iTCorrLim) { mTCorrLim = iTCorrLim; }
void FiniteElementMethodThermalCartesian2DSolver::setTBigCorr(double iTBigCorr) { mTBigCorr = iTBigCorr; }
void FiniteElementMethodThermalCartesian2DSolver::setBigNum(double iBigNum) { mBigNum = iBigNum; }
void FiniteElementMethodThermalCartesian2DSolver::setTAmb(double iTAmb) { mTAmb = iTAmb; }

int FiniteElementMethodThermalCartesian2DSolver::getLoopLim() { return mLoopLim; }
double FiniteElementMethodThermalCartesian2DSolver::getTCorrLim() { return mTCorrLim; }
double FiniteElementMethodThermalCartesian2DSolver::getTBigCorr() { return mTBigCorr; }
double FiniteElementMethodThermalCartesian2DSolver::getBigNum() { return mBigNum; }
double FiniteElementMethodThermalCartesian2DSolver::getTAmb() { return mTAmb; }

void FiniteElementMethodThermalCartesian2DSolver::setNodes()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting nodes...");

    size_t tNo = 1; // node number

    Node2D* tpN = NULL;

    for(plask::RectilinearMesh2D::iterator vec_it = mesh->begin(); vec_it != mesh->end(); ++vec_it) // loop through all nodes given in the correct iteration order
    {
        double x = vec_it->ee_x();
        double y = vec_it->ee_y();

        std::size_t i = vec_it.getIndex();
        auto it = mTConst.includes(*mesh, i);
        if (it != mTConst.end())
            tpN = new Node2D(tNo, x, y, it->condition, true);
        else
            tpN = new Node2D(tNo, x, y, mTAmb, false);

        mNodes.push_back(*tpN);

        delete tpN;
        tNo++;
    }
}

void FiniteElementMethodThermalCartesian2DSolver::setElements()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting elememts...");

    size_t tNo = 1; // element number

    Element2D* tpE = NULL;

    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    for(std::size_t i = 0; i < mesh->majorAxis().size()-1; i++)
    {
        for(std::size_t j = 0; j < mesh->minorAxis().size()-1; j++)
        {
            if (mesh->getIterationOrder() == 0) // more nodes in y-direction
                tpE = new Element2D(tNo, &(*ttN), &(*(ttN+1)), &(*(ttN+(mesh->minorAxis().size()))), &(*(ttN+(mesh->minorAxis().size()+1))));
            else // (mesh->getIterationOrder() == 1)
                tpE = new Element2D(tNo, &(*ttN), &(*(ttN+(mesh->minorAxis().size()))), &(*(ttN+1)), &(*(ttN+(mesh->minorAxis().size()+1))));
            tpE->setT();
            mElements.push_back(*tpE);

            delete tpE;
            tNo++;
            ttN++;
        }
        ttN++;
    }
}

void FiniteElementMethodThermalCartesian2DSolver::setHeatDensities()
{
    auto iMesh = mesh->getMidpointsMesh();
    try
    {
        mHeatDensities = inHeatDensity(iMesh);
    }
    catch (NoValue)
    {
        mHeatDensities.reset(iMesh->size(), 0.);
    }
}

void FiniteElementMethodThermalCartesian2DSolver::setSolver()
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

    mAWidth = mesh->minorAxis().size() + 3;
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

void FiniteElementMethodThermalCartesian2DSolver::delSolver()
{
    if (mLogs)
        writelog(LOG_INFO, "Deleting solver...");

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mTCorr.clear();
}

void FiniteElementMethodThermalCartesian2DSolver::setMatrix()
{
    if (mLogs)
        writelog(LOG_INFO, "Setting matrix...");

    std::vector<Element2D>::const_iterator ttE = mElements.begin();
    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    size_t tLoLeftNo, tLoRightNo, tUpLeftNo, tUpRightNo, // nodes numbers in current element
           tFstArg, tSecArg; // assistant values

    double tKXAssist, tKYAssist, tElemWidth, tElemHeight, tF, // assistant values to set stiffness matrix
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
        std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes(); // geometry->extract(GeometryObject::PredicateHasClass("active"));
        Vec<2, double> tSize;
        for (Box2D tBox: tVecBox)
        {
            if (tBox.includes(vec(ttE->getX(), ttE->getY())))
            {
                tSize = tBox.size();
                break;
            }
        }
        tKXAssist = geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first; // TODO
        tKYAssist = geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second; // TODO

        // set load vector
        tF = 0.25 * tElemWidth * tElemHeight * 1e-12 * mHeatDensities[ttE->getNo()-1]; // 1e-12 -> to transform um*um into m*m
        // heat per unit volume or heat rate per unit volume [W/(m^3)]

        // set symetric matrix components
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) / 6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/ 6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/ 6.;

        writelog(LOG_INFO, "%1% %2% %3% %4% %5% %6%", tK44, tK43, tK42, tK32, tF, mBigNum);

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
        mpA[tLoLeftNo-1][mAWidth-1]  += tF;
        mpA[tLoRightNo-1][mAWidth-1] += tF;
        mpA[tUpRightNo-1][mAWidth-1] += tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += tF;
    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getT()*mBigNum;
        }
}

void FiniteElementMethodThermalCartesian2DSolver::runCalc()
{
    //if (!isInitialized()) std::cout << "First calc.\n";
    ///else "Cont. calc.\n";
    initCalculation();
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

        tInfo = solveMatrix(mpA, mNodes.size(), mesh->minorAxis().size()+2);
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
            writelog(LOG_INFO, "Loop no: %1%(%2%), max. corr. for T: %3%", tLoop, mLoopNo, tTCorr);

            tLoop++;
        }
        else if (tInfo < 0)
            writelog(LOG_ERROR, "Wrong value of new temperature");
        else
            writelog(LOG_ERROR, "Wrong solver matrix");
    }

    //showNodes();

    saveTemperatures();

    if (mLogs)
        showTemperatures();

    delSolver();

    if (mLogs)
        writelog(LOG_INFO, "Temperature calculations completed");

    outTemperature.fireChanged();
    outHeatFlux.fireChanged();
}

void FiniteElementMethodThermalCartesian2DSolver::loadParam(const std::string &param, XMLReader &source, Manager &manager)
{
    if (param == "Tconst")
        manager.readBoundaryConditions(source,mTConst);
    if (param == "Tamb")
    {
        mTAmb = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
    if (param == "looplim")
    {
        mLoopLim = source.requireAttribute<int>("value");
        source.requireTagEnd();
    }
    if (param == "Tcorrlim")
    {
        mTCorrLim = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
    if (param == "Tbigcorr")
    {
        mTBigCorr = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
    if (param == "bignum")
    {
        mBigNum = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
    if (param == "logs")
    {
        mLogs = source.requireAttribute<bool>("value");
        source.requireTagEnd();
    }
}

void FiniteElementMethodThermalCartesian2DSolver::updNodes()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating nodes...");

    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        mTCorr[ttN->getNo()-1] = fabs( ttN->getT() - mpA[ttN->getNo()-1][mAWidth-1] ); // calculate corrections
        if (!ttN->ifTConst()) {
            //writelog(LOG_INFO, "mpA = %1%", mpA[ttN->getNo()-1][mAWidth-1]);
            ttN->setT( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of temperature
        }
        ttN++;
    }
}

void FiniteElementMethodThermalCartesian2DSolver::updElements()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating elements...");

    for (std::vector<Element2D>::iterator ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        ttE->setT();
}

void FiniteElementMethodThermalCartesian2DSolver::showNodes()
{
    writelog(LOG_INFO, "Showing nodes...");

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", x: " << ttN->getX() << ", y: " << ttN->getY() << ", T: " << ttN->getT() << std::endl; // TEST
}

void FiniteElementMethodThermalCartesian2DSolver::showElements()
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

void FiniteElementMethodThermalCartesian2DSolver::saveTemperatures()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving temperatures...");

    std::vector<Node2D>::const_iterator ttN;

    mTemperatures.reset(mNodes.size());

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN) {
        writelog(LOG_INFO, "getT = %1% [const %2%]", ttN->getT(), ttN->ifTConst());
        mTemperatures[ttN->getNo()-1] = ttN->getT();
    }
}

void FiniteElementMethodThermalCartesian2DSolver::saveHeatFluxes()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving heat fluxes...");

    std::vector<Element2D>::const_iterator ttE;

    mHeatFluxes.reset(mElements.size());

    std::vector<Box2D> tVecBox = geometry->getLeafsBoundingBoxes();

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
            - (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_x()).first) * ttE->getdTdX(),
            - (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->thermCond(ttE->getT(), tSize.ee_y()).second) * ttE->getdTdY() );
    }
}

void FiniteElementMethodThermalCartesian2DSolver::showTemperatures()
{
    std::cout << "Showing temperatures... " << mTemperatures<< std::endl;
}

int FiniteElementMethodThermalCartesian2DSolver::solveMatrix(double **ipA, long iN, long iBandWidth)
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

DataVector<const double> FiniteElementMethodThermalCartesian2DSolver::getTemperatures(const MeshD<2> &dst_mesh, InterpolationMethod method) const {
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*mesh, mTemperatures, dst_mesh, method);
}

DataVector<const Vec<2> > FiniteElementMethodThermalCartesian2DSolver::getHeatFluxes(const MeshD<2> &dst_mesh, InterpolationMethod method) const {
    if (method == DEFAULT_INTERPOLATION) method = INTERPOLATION_LINEAR;
    return interpolate(*mesh, mHeatFluxes, dst_mesh, method);
}

}}} // namespace plask::solvers::thermal
