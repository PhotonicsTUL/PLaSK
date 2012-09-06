#include "femV.h"

namespace plask { namespace solvers { namespace electrical {

FiniteElementMethodElectricalCartesian2DSolver::FiniteElementMethodElectricalCartesian2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    mLoopLim(5), mCorrLim(0.1), mBigNum(1e15) // TODO: it should be loaded from file
{
    mNodes.clear();
    mElements.clear();
    mPotentials.clear();
}

FiniteElementMethodElectricalCartesian2DSolver::~FiniteElementMethodElectricalCartesian2DSolver()
{
    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
}

void FiniteElementMethodElectricalCartesian2DSolver::setNodes()
{   
    std::cout << "Setting nodes...\n" << std::endl;

    size_t tNo = 1; // node number

    Node2D* tpN = NULL;

    for(plask::RectilinearMesh2D::iterator vec_it = mesh->begin(); vec_it != mesh->end(); ++vec_it) // loop through all nodes given in the correct iteration order
    {
        double x = vec_it->ee.x;
        double y = vec_it->ee.y;

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
            //tpE->setT();
            mElements.push_back(*tpE);

            delete tpE;
            tNo++;
            ttN++;
        }
        ttN++;
    }
}

void FiniteElementMethodElectricalCartesian2DSolver::setSolver()
{
    std::cout << "Setting solver...\n" << std::endl;

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
    mVcorr.clear();
    for(int i = 0; i < mAHeight; i++)
        mVcorr.push_back(0.);

    std::cout << "Main matrix width: " << mAWidth << std::endl; // TEST
    std::cout << "Main matrix height: " << mAHeight << std::endl; // TEST
}

void FiniteElementMethodElectricalCartesian2DSolver::delSolver()
{
    std::cout << "Deleting solver...\n" << std::endl;

    for (int i = 0; i < mAHeight; i++)
        delete [] mpA[i];
    delete [] mpA;
    mpA = NULL;

    mElements.clear();
    mNodes.clear();
    mVcorr.clear();
}

void FiniteElementMethodElectricalCartesian2DSolver::setMatrix()
{
    std::cout << "Setting matrix...\n" << std::endl;

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
        //double tX = 5.;
        //double tY = 5.;
        //double tT = 300.;
        //this->geometry->getMaterial(vec(tX, tY))->condT(tT);
        tKXAssist = /*geometry->getMaterial*/44.0; //!!! // TODO: call to database
        tKYAssist = 50000.0; //!!! // TODO: call to database (electrical conductivity)

        // set sources
        double tSource = 0.;
        tF = 0.25 * tElemWidth * tElemHeight * 1e24 * (cPhys::q) * (tSource); // -- give + // = 0 (If no heat-sources) //TODO: what should be here for real?

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
        if ( ttN->ifVConst() ) // TODO: Th night
        {
            //mpA[ttN->getNo()-1][mAWidth-2] += mBigNum; // this line is for the case when only corrections (dV for example) are calculated (used in DDM)
            mpA[ttN->getNo()-1][mAWidth-2] += mBigNum; // TODO: Th night
            mpA[ttN->getNo()-1][mAWidth-1] += ttN->getV()*mBigNum; // TODO: Th night
        }
}

void FiniteElementMethodElectricalCartesian2DSolver::runCalc()
{
    std::cout << "\nIt is time for Electrical Model!\n" << std::endl;

    //writelog(LOG_INFO, "It is time for Electrical Model!");

    //delSolver();
    setNodes();
    setElements();
    setSolver();

    std::cout << "Running solver...\n" << std::endl;

    std::vector<double>::const_iterator ttVcorr;
    int tLoop = mLoopLim, tInfo = 1;
    double tVcorr = 1e5; // much bigger than calculated corrections

    while (tLoop /*&& tVcorr > mCorrLim*/)
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
                tB.at(i) = mpA[i][mAWidth-1]; // mpA[i][mAWidth-1] - here are new values of potential

            //ttVcorr = std::max_element(mVcorr.begin(), mVcorr.end());

            //tVcorr = *ttVcorr;

            // show max correction
            //std::cout << "Max corr. = " << tVcorr << "\n";
            --tLoop;
        }
        else if (tInfo < 0)
            std::cout << "Wrong value of new V.\n";
        else
            std::cout << "Wrong solver matrix.\n";
    }

    showNodes();

    savePote();

    showPote();

    delSolver();

    std::cout << "Potential calculations completed\n" << std::endl;
}

void FiniteElementMethodElectricalCartesian2DSolver::loadParam(const std::string &param, XMLReader &source, Manager &manager)
{
    if (param == "Vconst")
        manager.readBoundaryConditions(source,mVconst);
}

void FiniteElementMethodElectricalCartesian2DSolver::updNodes()
{
    std::cout << "Updating nodes...\n" << std::endl;

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
    std::cout << "Updating elements...\n" << std::endl;

    //for (std::vector<Element2D>::iterator ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
    //    ttE->setT(); // TODO
}

void FiniteElementMethodElectricalCartesian2DSolver::showNodes()
{
    std::cout << "Showing nodes...\n" << std::endl;

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", x: " << ttN->getX() << ", y: " << ttN->getY() << ", T: " << ttN->getV() << std::endl; // TEST
}

void FiniteElementMethodElectricalCartesian2DSolver::showElements()
{
    std::cout << "Showing elements...\n" << std::endl;

    std::vector<Element2D>::const_iterator ttE;

    for (ttE = mElements.begin(); ttE != mElements.end(); ++ttE)
        std::cout << "Element no: " << ttE->getNo()
                     << ", BL: " << ttE->getNLoLeftPtr()->getNo() << " (" << ttE->getNLoLeftPtr()->getX() << "," << ttE->getNLoLeftPtr()->getY() << ")"
                     << ", BR: " << ttE->getNLoRightPtr()->getNo() << " (" << ttE->getNLoRightPtr()->getX() << "," << ttE->getNLoRightPtr()->getY() << ")"
                     << ", TL: " << ttE->getNUpLeftPtr()->getNo() << " (" << ttE->getNUpLeftPtr()->getX() << "," << ttE->getNUpLeftPtr()->getY() << ")"
                     << ", TR: " << ttE->getNUpRightPtr()->getNo() << " (" << ttE->getNUpRightPtr()->getX() << "," << ttE->getNUpRightPtr()->getY() << ")"
                     << std::endl; // TEST
}

void FiniteElementMethodElectricalCartesian2DSolver::savePote()
{
    std::cout << "Saving potentials...\n" << std::endl;

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        mPotentials.push_back(ttN->getV());
}

void FiniteElementMethodElectricalCartesian2DSolver::showPote()
{
    std::cout << "Showing potentials...\n" << std::endl;

    std::vector<Node2D>::const_iterator ttN;

    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        std::cout << "Node no: " << ttN->getNo() << ", V: " << mPotentials[ttN->getNo()-1] << std::endl;
}

int FiniteElementMethodElectricalCartesian2DSolver::solveMatrix(double **ipA, long iN, long iBandWidth)
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

}}} // namespace plask::solvers::eletrical
