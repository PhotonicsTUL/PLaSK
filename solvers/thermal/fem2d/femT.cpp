#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

FiniteElementMethodThermal2DSolver::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(100), mCorrLim(0.1), mBigNum(1e15) // TODO: it should be loaded from file
{
}

FiniteElementMethodThermal2DSolver::~FiniteElementMethodThermal2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

void FiniteElementMethodThermal2DSolver::setSolver()
{
    std::cout << "Setting solver\n" << std::endl;
    size_t tSize = 0;

    size_t tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
    size_t tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (larger one)

    if (tNoOfNodesX > tNoOfNodesY) tSize = tNoOfNodesY + 2;
    else tSize = tNoOfNodesX + 2;

    mAWidth = tSize + 1;
    mAHeight = mNodes.size();
    mpA = new double*[mAHeight];
    for(int i = 0; i < mAHeight; i++)
        mpA[i] = new double[mAWidth];
    mTcorr.clear();
    for(int i = 0; i < mAHeight; i++)
        mTcorr.push_back(0.);
}

void FiniteElementMethodThermal2DSolver::delSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mElements.clear();
    mNodes.clear();
    mTcorr.clear();
}

void FiniteElementMethodThermal2DSolver::setMatrix()
{
    std::vector<Element2D>::const_iterator ttE = mElements.begin();
    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    size_t tLoLeftNo = 0, tLoRightNo = 0, tUpLeftNo = 0, tUpRightNo = 0, // nodes numbers in current element
           tFstArg = 0, tSecArg = 0; // assistant values

    double tValLoLeft = 0., tValLoRight = 0., tValUpLeft = 0., tValUpRight = 0., // assistant values of nodes parameters
           tKXAssist = 0., tKYAssist = 0., tElemWidth = 0., tElemHeight = 0., tF = 0., // assistant values to set stiffness matrix
           tK11 = 0., tK21 = 0., tK31 = 0., tK41 = 0., tK22 = 0., tK32 = 0., tK42 = 0., tK33 = 0., tK43 = 0., tK44 = 0.; // local symetric matrix components

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
        tKXAssist = 44.0; //!!! // TODO: call to database
        tKYAssist = 44.0; //!!! // TODO: call to database

        // set sources
        double tHSJoule = 0.;
        tF = 0.25 * tElemWidth * tElemHeight * 1e24 * (cPhys::q) * (tHSJoule); // -- give + // = 0 (If no heat-sources)

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

        // get temperatures from previous step
        tValLoLeft = ttE->getNLoLeftPtr()->getT();
        tValLoRight = ttE->getNLoRightPtr()->getT();
        tValUpLeft = ttE->getNUpLeftPtr()->getT();
        tValUpRight = ttE->getNUpRightPtr()->getT();

        // set load vector
        mpA[tLoLeftNo-1][mAWidth-1]  += -(tK11*tValLoLeft + tK21*tValLoRight + tK31*tValUpRight + tK41*tValUpLeft) + tF;
        mpA[tLoRightNo-1][mAWidth-1] += -(tK21*tValLoLeft + tK22*tValLoRight + tK32*tValUpRight + tK42*tValUpLeft) + tF;
        mpA[tUpRightNo-1][mAWidth-1] += -(tK31*tValLoLeft + tK32*tValLoRight + tK33*tValUpRight + tK43*tValUpLeft) + tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += -(tK41*tValLoLeft + tK42*tValLoRight + tK43*tValUpRight + tK44*tValUpLeft) + tF;

    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() )
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
}

void FiniteElementMethodThermal2DSolver::runCalc()
{
    std::cout << "Setting solver...\n";
    delSolver();
    setSolver();
    std::cout << "Done.\n";
    std::cout << "Running solver...\n" << std::endl;

    std::vector<double>::const_iterator ttTcorr;
    int tLoop = mLoopLim, tInfo = 1;
    double tTcorr = 100.; // much bigger than calculated corrections

    size_t tSize = 0;

    size_t tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
    size_t tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (larger one)

    if (tNoOfNodesX > tNoOfNodesY) tSize = tNoOfNodesY + 1;
    else tSize = tNoOfNodesX + 1;

    while (tLoop && tTcorr > mCorrLim)
    {
        setMatrix();

        tInfo = solve3Diag(mpA, mNodes.size(), tSize+1);
        if (!tInfo)
        {
            // update
            updNodes();
            updElements();

            // find max correction
            std::vector<double> tB(mAHeight, 0.);

            for (int i = 0; i < mAHeight; i++)
                tB.at(i) = mpA[i][mAWidth-1]; // mpA[i][mAWidth-1] - here are new values of temperature

            ttTcorr = std::max_element(mTcorr.begin(), mTcorr.end());

            tTcorr = *ttTcorr;

            // show max correction
            std::cout << "Max corr. = " << tTcorr << "\n";
            --tLoop;
        }
        else if (tInfo < 0) std::cout << "Wrong value of new T.\n";
        else std::cout << "Wrong solver matrix.\n";
    }

    std::cout << "Temperature calculations completed\n" << std::endl;
}

/// Update nodes
void FiniteElementMethodThermal2DSolver::updNodes()
{
    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        mTcorr[ttN->getNo()-1] = fabs( ttN->getT() - mpA[ttN->getNo()-1][mAWidth-1] ); // calculate corrections
        if (!ttN->ifTConst())
            ttN->setT( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of temperature
        ttN++;
    }
}

/// Update elements
void FiniteElementMethodThermal2DSolver::updElements()
{
    for (std::vector<Element2D>::iterator ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        ttE->setT();
}

/// Solve 3-diag matrix
int FiniteElementMethodThermal2DSolver::solve3Diag(double **ipA, long iN, long iBandWidth)
{
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

}}} // namespace plask::solvers::finiteT
