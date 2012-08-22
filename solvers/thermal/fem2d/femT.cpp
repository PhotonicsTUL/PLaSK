#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

FiniteElementMethodThermal2DSolver::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(100), mCorrLim(0.1), mBigNum(1e15)
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
    std::cout << "Setting solver" << std::endl; //TEST
    size_t tSize = 0;

    size_t tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
    size_t tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (larger one)

    if (tNoOfNodesX > tNoOfNodesY) tSize = tNoOfNodesY + 2;
    else tSize = tNoOfNodesX + 2;

    mAWidth = tSize+1;
    mAHeight = mNodes.size();
    mpA = new double*[mAHeight];
    for(int i=0; i<mAHeight; i++)
        mpA[i]=new double[mAWidth];
}

void FiniteElementMethodThermal2DSolver::delSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;

    mpA = NULL;
}

void FiniteElementMethodThermal2DSolver::setMatrix()
{
    std::vector<Element2D>::const_iterator ttE = mElements.begin();
    std::vector<Node2D>::const_iterator ttN = mNodes.begin();

    size_t tLoLeftNo = 0, tLoRightNo = 0, tUpLeftNo = 0, tUpRightNo = 0, //Element Nodes Nos
           tFstArg = 0, tSecArg = 0; //Assistant Values To Set Vector A

    double tValLoLeft = 0., tValLoRight = 0., tValUpLeft = 0., tValUpRight = 0., //Assistant Values of Nodes Parameters
           tKXAssist = 0., tKYAssist = 0., tElemWidth = 0., tElemHeight = 0., tF = 0., //Assistant Values to Set K and G Components
           tK11 = 0., tK21 = 0., tK31 = 0., tK41 = 0., tK22 = 0., tK32 = 0., tK42 = 0., tK33 = 0., tK43 = 0., tK44 = 0.; //Local Symetric Matrix Components

    //Set Data Zeros
    for(int i=0; i<mAHeight; i++)
        for(int j=0; j<mAWidth; j++)
            mpA[i][j]=0.;

    //Set Vector A And Vector B
    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    {
        //Set Element Nodes Nos
        tLoLeftNo = ttE->getNLoLeftPtr()->getNo();
        tLoRightNo = ttE->getNLoRightPtr()->getNo();
        tUpLeftNo = ttE->getNUpLeftPtr()->getNo();
        tUpRightNo = ttE->getNUpRightPtr()->getNo();

        //Set Elements Size
        tElemWidth = fabs(ttE->getNLoLeftPtr()->getX() - ttE->getNLoRightPtr()->getX());
        tElemHeight = fabs(ttE->getNLoLeftPtr()->getY() - ttE->getNUpLeftPtr()->getY());

        //Set Components (K, G, F)

        tKXAssist = 44.0; //!!! // TODO: call to database
        tKYAssist = 44.0; //!!! // TODO: call to database

        //tF
        double tHJoule = 0.;
        tF = 0.25*tElemWidth*tElemHeight*1e24*(cPhys::Q) * (tHJoule); // -- give + // = 0 (If no heat-sources)

        //Calculating K, G
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) /6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/6.;

        //Set Matrix A
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

        //Get Element Nodes Values (Psi,Fn,Fp)
        tValLoLeft = ttE->getNLoLeftPtr()->getT();
        tValLoRight = ttE->getNLoRightPtr()->getT();
        tValUpLeft = ttE->getNUpLeftPtr()->getT();
        tValUpRight = ttE->getNUpRightPtr()->getT();

        //Set Vector B
        mpA[tLoLeftNo-1][mAWidth-1]  += -(tK11*tValLoLeft + tK21*tValLoRight + tK31*tValUpRight + tK41*tValUpLeft) + tF;
        mpA[tLoRightNo-1][mAWidth-1] += -(tK21*tValLoLeft + tK22*tValLoRight + tK32*tValUpRight + tK42*tValUpLeft) + tF;
        mpA[tUpRightNo-1][mAWidth-1] += -(tK31*tValLoLeft + tK32*tValLoRight + tK33*tValUpRight + tK43*tValUpLeft) + tF;
        mpA[tUpLeftNo-1][mAWidth-1]  += -(tK41*tValLoLeft + tK42*tValLoRight + tK43*tValUpRight + tK44*tValUpLeft) + tF;

    }
    //Add Big Number to Data A (//if (ttN->getVolContSide() != 'M' || ttN->ifTConst()))
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() )
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum;
}

/// Run single temperature calculations
void FiniteElementMethodThermal2DSolver::runCalc()
{
    std::vector<double>::const_iterator ttCorrPos, ttCorrNeg;
    int tLoop = mLoopLim, tInfo = 1;
    double tCorr = 100.;

    size_t tSize = 0;

    size_t tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (smaller one)
    size_t tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (larger one)

    if (tNoOfNodesX > tNoOfNodesY) tSize = tNoOfNodesY + 1;
    else tSize = tNoOfNodesX + 1;

    while (tLoop && tCorr > mCorrLim)
    {
        //Solve Part
        setMatrix();

        tInfo = solve3Diag(mpA, mNodes.size(), tSize+1);
        if (!tInfo)
        {
            //Update Model
            updNodes();
            updElements();
            //Find Max Correction
            std::vector<double> tB(mAHeight, 0.);

            for (int i = 0; i < mAHeight; i++)
                tB.at(i) = mpA[i][mAWidth-1];

            ttCorrPos = std::max_element(tB.begin(), tB.end());
            ttCorrNeg = std::min_element(tB.begin(), tB.end());

            fabs(*ttCorrNeg) > *ttCorrPos ? tCorr = fabs(*ttCorrNeg) : tCorr = *ttCorrPos;

            //Show Max Correction
            std::cout << "Max corr. = " << tCorr << "\n";
            --tLoop;
        }
        else if (tInfo < 0) std::cout << "Wrong value of new T\n";
        else std::cout << "Wrong solver matrix\n";
    }
}

/// Find max correction for temperature
double FiniteElementMethodThermal2DSolver::findMaxCorr()
{
    return 0.;
}

/// Update nodes
void FiniteElementMethodThermal2DSolver::updNodes()
{
    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        if (!ttN->ifTConst())
            ttN->setT( mpA[ttN->getNo()-1][mAWidth-1] );
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
int FiniteElementMethodThermal2DSolver::solve3Diag(double **a, long n, long SZER_PASMA)
{
    //std::cout << "\nWTF\n";
    long m, k, i, j, poc, kon, q, mn;
    double SUM;

    //n = li_x * li_y;	//wielkosc macierzy

    //utworzenie macierzy L
    for( j = 0; j < n; j++)
    {
        poc = j - (SZER_PASMA-1);
        if( poc < 0 ) poc = 0;
        kon = j + (SZER_PASMA-1);
        if( kon > (n-1) ) kon = n-1;
        SUM = 0.0;
        for( k = poc; k <= (j-1); k++)
        {
            m = k - j + (SZER_PASMA-1);
            SUM += a[j][m]*a[j][m];
        }
        m = j - j + (SZER_PASMA-1);
        a[j][m] = sqrt(a[j][m] - SUM);
        for( i = j+1; i <= kon; i++)
        {
            if( i > (SZER_PASMA-1)) mn = 1;
            else					mn = 0;
            poc += 1*mn;
            SUM = 0.0;
            for( k = poc; k <= (j-1); k++)
            {
                m = k - i + (SZER_PASMA-1);
                q = k - j + (SZER_PASMA-1);
                SUM += a[i][m]*a[j][q];
            }
            m = j - i + (SZER_PASMA-1);
            q = j - j + (SZER_PASMA-1);
            a[i][m] = ( a[i][m] - SUM ) / a[j][q];
        }
    }

    //rozwiazanie ukladu LY=B
    for( j = 0; j < n; j++)
    {
        poc = j - (SZER_PASMA-1);
        if( poc < 0 ) poc = 0;
        SUM = 0.0;
        for( k = poc; k <= (j-1); k++)
        {
            m = k - j + (SZER_PASMA-1);
            SUM += a[j][m]*a[k][SZER_PASMA];
        }
        m = j - j + (SZER_PASMA-1);
        a[j][SZER_PASMA] = ( a[j][SZER_PASMA] - SUM ) / a[j][m];
    }

    //rozwiazanie ukladu L^TX=Y
    for( j = (n-1); j >= 0; j--)
    {
        kon = j + (SZER_PASMA-1);
        if( kon > (n-1) ) kon = n-1;
        SUM = 0.0;
        for( k = j+1; k <= kon; k++)
        {
            m = j - k + (SZER_PASMA-1);
            SUM += a[k][m]*a[k][SZER_PASMA];
        }
        m = j - j + (SZER_PASMA-1);
        a[j][SZER_PASMA] = ( a[j][SZER_PASMA] - SUM ) / a[j][m];
    }
    //std::cout << "\nWTF2\n";
    return 0;
}

}}} // namespace plask::solvers::finiteT
