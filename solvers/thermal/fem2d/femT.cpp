#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

FiniteElementMethodThermalCartesian2DSolver::FiniteElementMethodThermalCartesian2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(5), mCorrLim(0.1), mBigNum(1e15) // TODO: it should be loaded from file
{
    mNodes.clear();
    mElements.clear();
}

FiniteElementMethodThermalCartesian2DSolver::~FiniteElementMethodThermalCartesian2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

void FiniteElementMethodThermalCartesian2DSolver::setNodes()
{   
    std::cout << "Setting nodes...\n" << std::endl;

    size_t tNo = 1; // node number

    Node2D* tpN = NULL;

    for(plask::RectilinearMesh2D::iterator vec_it = mesh->begin(); vec_it != mesh->end(); ++vec_it) // loop through all nodes given in the correct iteration order
    {
        double x = vec_it->ee.x;
        double y = vec_it->ee.y;

        std::size_t i = vec_it.getIndex();
        auto it = mTconst.includes(*mesh, i);
        if (it != mTconst.end())
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
    std::cout << "Setting elememts...\n" << std::endl;

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

void FiniteElementMethodThermalCartesian2DSolver::setSolver()
{
    std::cout << "Setting solver...\n" << std::endl;

    size_t tSize = 0;

    size_t tNoOfNodesX, tNoOfNodesY;

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

    std::cout << "NoOfNodesX: " << tNoOfNodesX << std::endl; // TEST
    std::cout << "NoOfNodesY: " << tNoOfNodesY << std::endl; // TEST

    size_t tNoOfNodes = mesh->size(); // number of all nodes
    std::cout << "NoOfNodes: " << tNoOfNodes << std::endl; // TEST

    mAWidth = mesh->minorAxis().size() + 3;
    mAHeight = mNodes.size();
    mpA = new double*[mAHeight];
    for(int i = 0; i < mAHeight; i++)
        mpA[i] = new double[mAWidth];
    mTcorr.clear();
    for(int i = 0; i < mAHeight; i++)
        mTcorr.push_back(0.);

    std::cout << "Main matrix width: " << mAWidth << std::endl; // TEST
    std::cout << "Main matrix height: " << mAHeight << std::endl; // TEST
}

void FiniteElementMethodThermalCartesian2DSolver::delSolver()
{
    std::cout << "Deleting solver...\n" << std::endl;

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mElements.clear();
    mNodes.clear();
    mTcorr.clear();
}

void FiniteElementMethodThermalCartesian2DSolver::setMatrix()
{
    std::cout << "Setting matrix...\n" << std::endl;

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
        //double tX = 5.;
        //double tY = 5.;
        //double tT = 300.;
        //this->geometry->getMaterial(vec(tX, tY))->condT(tT);
        tKXAssist = /*geometry->getMaterial*/44.0; //!!! // TODO: call to database
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
        mpA[tLoLeftNo-1][mAWidth-1]  += -0*(tK11*tValLoLeft + tK21*tValLoRight + tK31*tValUpRight + tK41*tValUpLeft) + tF; // TODO: Th night
        mpA[tLoRightNo-1][mAWidth-1] += -0*(tK21*tValLoLeft + tK22*tValLoRight + tK32*tValUpRight + tK42*tValUpLeft) + tF; // TODO: Th night
        mpA[tUpRightNo-1][mAWidth-1] += -0*(tK31*tValLoLeft + tK32*tValLoRight + tK33*tValUpRight + tK43*tValUpLeft) + tF; // TODO: Th night
        mpA[tUpLeftNo-1][mAWidth-1]  += -0*(tK41*tValLoLeft + tK42*tValLoRight + tK43*tValUpRight + tK44*tValUpLeft) + tF; // TODO: Th night

    }
    // boundary conditions are taken into account
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ( ttN->ifTConst() ) // TODO: Th night
        {
            //mpA[ttN->getNo()-1][mAWidth-2] += mBigNum; // this line is for the case when only corrections (dT for example) are calculated (used in DDM)
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum; // TODO: Th night
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getT()*mBigNum; // TODO: Th night
        }
}

void FiniteElementMethodThermalCartesian2DSolver::runCalc()
{
    std::cout << "\nIt is time for Thermal Model!\n" << std::endl;

    //writelog(LOG_INFO, "It is time for Thermal Model!");

    //delSolver();
    setNodes();
    setElements();
    setSolver();

    std::cout << "Running solver...\n" << std::endl;

    std::vector<double>::const_iterator ttTcorr;
    int tLoop = mLoopLim, tInfo = 1;
    double tTcorr = 1e5; // much bigger than calculated corrections

    while (tLoop /*&& tTcorr > mCorrLim*/)
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

            //ttTcorr = std::max_element(mTcorr.begin(), mTcorr.end());

            //tTcorr = *ttTcorr;

            // show max correction
            //std::cout << "Max corr. = " << tTcorr << "\n";
            --tLoop;
        }
        else if (tInfo < 0)
            std::cout << "Wrong value of new T.\n";
        else
            std::cout << "Wrong solver matrix.\n";
    }

    showNodes();

    delSolver();

    std::cout << "Temperature calculations completed\n" << std::endl;
}

void FiniteElementMethodThermalCartesian2DSolver::loadParam(const std::string &param, XMLReader &source, Manager &manager)
{
    if (param == "Tconst")
        manager.readBoundaryConditions(source,mTconst);
    if (param == "Tamb") {
        mTAmb = source.requireAttribute<double>("value");
        source.requireTagEnd();
    }
}

void FiniteElementMethodThermalCartesian2DSolver::updNodes()
{
    std::cout << "Updating nodes...\n" << std::endl;

    std::vector<Node2D>::iterator ttN = mNodes.begin();

    while (ttN != mNodes.end())
    {
        mTcorr[ttN->getNo()-1] = fabs( ttN->getT() - mpA[ttN->getNo()-1][mAWidth-1] ); // calculate corrections
        if (!ttN->ifTConst())
            ttN->setT( mpA[ttN->getNo()-1][mAWidth-1] ); // mpA[ttN->getNo()-1][mAWidth-1] - here are new values of temperature
        ttN++;
    }
}

void FiniteElementMethodThermalCartesian2DSolver::updElements()
{
    std::cout << "Updating elements...\n" << std::endl;

    for (std::vector<Element2D>::iterator ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        ttE->setT();
}

void FiniteElementMethodThermalCartesian2DSolver::showNodes()
{
    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", x: " << ttN->getX() << ", y: " << ttN->getY() << ", T: " << ttN->getT() << std::endl; // TEST
}

void FiniteElementMethodThermalCartesian2DSolver::showElements()
{
    std::vector<Element2D>::const_iterator ttE;

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        std::cout << "Element no: " << ttE->getNo()
                     << ", BL: " << ttE->getNLoLeftPtr()->getNo() << " (" << ttE->getNLoLeftPtr()->getX() << "," << ttE->getNLoLeftPtr()->getY() << ")"
                     << ", BR: " << ttE->getNLoRightPtr()->getNo() << " (" << ttE->getNLoRightPtr()->getX() << "," << ttE->getNLoRightPtr()->getY() << ")"
                     << ", TL: " << ttE->getNUpLeftPtr()->getNo() << " (" << ttE->getNUpLeftPtr()->getX() << "," << ttE->getNUpLeftPtr()->getY() << ")"
                     << ", TR: " << ttE->getNUpRightPtr()->getNo() << " (" << ttE->getNUpRightPtr()->getX() << "," << ttE->getNUpRightPtr()->getY() << ")"
                     << std::endl; // TEST
}

int FiniteElementMethodThermalCartesian2DSolver::solveMatrix(double **ipA, long iN, long iBandWidth)
{
    std::cout << "Solving matrix...\n" << std::endl;

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
