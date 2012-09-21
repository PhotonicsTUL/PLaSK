#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

FiniteElementMethodElectricalCartesian2DSolver::FiniteElementMethodElectricalCartesian2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(5),
    mVCorrLim(0.01),
    mVBigCorr(1e5),
    mBigNum(1e15),
    mLogs(false),
    mLoopNo(0),
    outPotential(this, &FiniteElementMethodElectricalCartesian2DSolver::getPotentials),
    outCurrentDensity(this, &FiniteElementMethodElectricalCartesian2DSolver::getCurrentDensities),
    outHeatDensity(this, &FiniteElementMethodElectricalCartesian2DSolver::getHeatDensities)
{
    mNodes.clear();
    mElements.clear();
    mPotentials.reset();
    mCurrentDensities.reset();
    mTemperatures.reset();
    mHeatDensities.reset();

    inTemperature = 300.;
}

FiniteElementMethodElectricalCartesian2DSolver::~FiniteElementMethodElectricalCartesian2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

void FiniteElementMethodElectricalCartesian2DSolver::onInitialize() // In this function check if geometry and mesh are set
{
    if (!geometry) throw NoGeometryException(getId());
    if (!mesh) throw NoMeshException(getId());
    setNodes();
    setElements();
}

void FiniteElementMethodElectricalCartesian2DSolver::onInvalidate() // This will be called when e.g. geometry or mesh changes and your results become outdated
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

void FiniteElementMethodElectricalCartesian2DSolver::setLoopLim(int iLoopLim) { mLoopLim = iLoopLim; }
void FiniteElementMethodElectricalCartesian2DSolver::setVCorrLim(double iVCorrLim) { mVCorrLim = iVCorrLim; }
void FiniteElementMethodElectricalCartesian2DSolver::setVBigCorr(double iVBigCorr) { mVBigCorr = iVBigCorr; }
void FiniteElementMethodElectricalCartesian2DSolver::setBigNum(double iBigNum) { mBigNum = iBigNum; }

int FiniteElementMethodElectricalCartesian2DSolver::getLoopLim() { return mLoopLim; }
double FiniteElementMethodElectricalCartesian2DSolver::getVCorrLim() { return mVCorrLim; }
double FiniteElementMethodElectricalCartesian2DSolver::getVBigCorr() { return mVBigCorr; }
double FiniteElementMethodElectricalCartesian2DSolver::getBigNum() { return mBigNum; }

void FiniteElementMethodElectricalCartesian2DSolver::setNodes()
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
        auto it = mVconst.includes(*mesh, i);
        if (it != mVconst.end())
            tpN = new Node2D(tNo, x, y, it->condition, true);
        else
            tpN = new Node2D(tNo, x, y, 0., false);

        mNodes.push_back(*tpN);

        delete tpN;
        tNo++;
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::setElements()
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
            //tpE->setT();
            mElements.push_back(*tpE);

            delete tpE;
            tNo++;
            ttN++;
        }
        ttN++;
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::setTemperatures()
{
    auto iMesh = mesh->getMidpointsMesh();
    try
    {
        mTemperatures = inTemperature(iMesh);
    }
    catch (NoValue)
    {
        mTemperatures.reset(iMesh->size(), 300.);
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::setSolver()
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
    mVcorr.clear();
    for(int i = 0; i < mAHeight; i++)
        mVcorr.push_back(mVBigCorr);

    //std::cout << "Main matrix width: " << mAWidth << std::endl; // TEST
    //std::cout << "Main matrix height: " << mAHeight << std::endl; // TEST
}

void FiniteElementMethodElectricalCartesian2DSolver::delSolver()
{
    if (mLogs)
        writelog(LOG_INFO, "Deleting solver...");

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mVcorr.clear();
}

void FiniteElementMethodElectricalCartesian2DSolver::setMatrix()
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
        tKXAssist = geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()]).first;
        tKYAssist = geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()]).second;

        // set load vector
        tF = 0.; //0.25 * tElemWidth * tElemHeight * 1e-12 * 0.; // 1e-12 -> to transform um*um into m*m

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
        mpA[tLoLeftNo-1][mAWidth-1]  += tF;
        mpA[tLoRightNo-1][mAWidth-1] += tF;
        mpA[tUpRightNo-1][mAWidth-1] += tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += tF;

    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifVConst() )
        {
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getV()*mBigNum;
        }
}

void FiniteElementMethodElectricalCartesian2DSolver::runCalc()
{
    //if (!isInitialized()) std::cout << "First calc.\n";
    ///else "Cont. calc.\n";
    initCalculation();
    //if (!inTemperatures.changed) return;

    if (mLogs)
        writelog(LOG_INFO, "Starting electrical calculations...");

    setSolver();

    setTemperatures();

    if (mLogs)
        writelog(LOG_INFO, "Running solver...");

    std::vector<double>::const_iterator ttVCorr;
    int tLoop = 1, tInfo = 1;
    double tVCorr = mVBigCorr; // much bigger than calculated corrections

    while ( (tLoop <= mLoopLim) && (tVCorr > mVCorrLim) )
    {
        setMatrix();

        tInfo = solveMatrix(mpA, mNodes.size(), mesh->minorAxis().size()+2);
        if (!tInfo)
        {
            // update
            updNodes();
            //updElements();

            // find max correction
            std::vector<double> tB(mAHeight, 0.);

            for (int i = 0; i < mAHeight; i++)
                tB.at(i) = mpA[i][mAWidth-1]; // mpA[i][mAWidth-1] - here are new values of potential

            ttVCorr = std::max_element(mVcorr.begin(), mVcorr.end());

            tVCorr = *ttVCorr;

            mLoopNo++;

            // show max correction
            writelog(LOG_INFO, "Loop no: %1%(%2%), max. corr. for V: %3%", tLoop, mLoopNo, tVCorr);

            tLoop++;
        }
        else if (tInfo < 0)
            writelog(LOG_ERROR, "Wrong value of new potential");
        else
            writelog(LOG_ERROR, "Wrong solver matrix");
    }

    //showNodes();

    savePotentials();

    saveCurrentDensities();

    saveHeatDensities();

    if (mLogs)
        showPotentials();

    delSolver();

    if (mLogs)
        writelog(LOG_INFO, "Potential calculations completed");

    outPotential.fireChanged();
    outCurrentDensity.fireChanged();
    outHeatDensity.fireChanged();
}

void FiniteElementMethodElectricalCartesian2DSolver::loadParam(const std::string &param, XMLReader &source, Manager &manager)
{
    if (param == "Vconst")
        manager.readBoundaryConditions(source,mVconst);
    if (param == "looplim")
    {
        mLoopLim = source.requireAttribute<int>("value");
        source.requireTagEnd();
    }
    if (param == "Vcorrlim")
    {
        mVCorrLim = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
    if (param == "Vbigcorr")
    {
        mVBigCorr = source.requireAttribute<double>("value");
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

void FiniteElementMethodElectricalCartesian2DSolver::updNodes()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating nodes...");

    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        mVcorr[ttN->getNo()-1] = fabs( ttN->getV() - mpA[ttN->getNo()-1][mAWidth-1] ); // calculate corrections
        if (!ttN->ifVConst())
            ttN->setV( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of potential
        ttN++;
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::updElements()
{
    if (mLogs)
        writelog(LOG_INFO, "Updating elements...");
}

void FiniteElementMethodElectricalCartesian2DSolver::showNodes()
{
    writelog(LOG_INFO, "Showing nodes...");

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", x: " << ttN->getX() << ", y: " << ttN->getY() << ", T: " << ttN->getV() << std::endl; // TEST
}

void FiniteElementMethodElectricalCartesian2DSolver::showElements()
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

void FiniteElementMethodElectricalCartesian2DSolver::savePotentials()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving potentials...");

    std::vector<Node2D>::const_iterator ttN;

    mPotentials.reset(mNodes.size());

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        mPotentials[ttN->getNo()-1] = ttN->getV();
}

void FiniteElementMethodElectricalCartesian2DSolver::saveCurrentDensities()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving current densities...");

    std::vector<Element2D>::const_iterator ttE;

    mCurrentDensities.reset(mElements.size());

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        mCurrentDensities[ttE->getNo()-1] = vec(
            - (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).first) * ttE->getdVdX(),
            - (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).second) * ttE->getdVdY() );
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::saveHeatDensities()
{
    if (mLogs)
        writelog(LOG_INFO, "Saving heat densities...");

    std::vector<Element2D>::const_iterator ttE;

    mHeatDensities.reset(mElements.size());

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        mHeatDensities[ttE->getNo()-1] = (mCurrentDensities[ttE->getNo()-1]).ee_x() * (mCurrentDensities[ttE->getNo()-1]).ee_x() / (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).first) +
            (mCurrentDensities[ttE->getNo()-1]).ee_y() * (mCurrentDensities[ttE->getNo()-1]).ee_y() / (geometry->getMaterial(vec(ttE->getX(), ttE->getY()))->cond(mTemperatures[ttE->getNo()-1]).second);
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::showPotentials()
{
    std::cout << "Showing potentials... " << mPotentials << std::endl;
}

int FiniteElementMethodElectricalCartesian2DSolver::solveMatrix(double **ipA, long iN, long iBandWidth)
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

DataVector<const double> FiniteElementMethodElectricalCartesian2DSolver::getPotentials(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*mesh, mPotentials, dst_mesh, method);
}

DataVector<const Vec<2> > FiniteElementMethodElectricalCartesian2DSolver::getCurrentDensities(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*(mesh->getMidpointsMesh()), mCurrentDensities, dst_mesh, method);
}

DataVector<const double> FiniteElementMethodElectricalCartesian2DSolver::getHeatDensities(const MeshD<2> &dst_mesh, InterpolationMethod method) const
{
    if (method == DEFAULT_INTERPOLATION)
        method = INTERPOLATION_LINEAR;
    return interpolate(*(mesh->getMidpointsMesh()), mHeatDensities, dst_mesh, method);
}

}}} // namespace plask::solvers::eletrical
