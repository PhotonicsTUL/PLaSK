#include "femT.h"

namespace plask { namespace solvers { namespace thermal {

FiniteElementMethodThermal2DSolver::FiniteElementMethodThermal2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DCartesian, RectilinearMesh2D>(name),
    maxiterations(10)
{
}

FiniteElementMethodThermal2DSolver::~FiniteElementMethodThermal2DSolver()
{
    delete mA;
    delete mB;
}

void FiniteElementMethodThermal2DSolver::setSolver()
{
    std::cout << "Setting solver" << std::endl;
    size_t tNoOfXPts = mesh->minorAxis().size(); // number of nodes on minor axis (for optimal mesh smaller one)
    size_t tSize = tNoOfXPts + 2;
    size_t tNoNodes = mesh->size(); // number of all the nodes
    mA = new std::vector<double> (tNoNodes*tSize, 0.);
    mB = new std::vector<double> (tNoNodes, 0.);
}

void FiniteElementMethodThermal2DSolver::delSolver()
{
    delete mA;
    delete mB;
    mA = NULL;
    mB = NULL;
}

void FiniteElementMethodThermal2DSolver::setMatrixData()
{
    //(*getMesh)[0].


    size_t tNoOfNodesX = mesh->minorAxis().size(); // number of nodes on minor axis (for optimal mesh smaller one)
    size_t tNoOfNodesY = mesh->majorAxis().size(); // number of nodes on major axis (for optimal mesh larger one)

    size_t tNoOfElements = (tNoOfNodesX -1) * (tNoOfNodesY -1); // number of elements (all)

    //getMesh()->setOptimalIterationOrder();
/*
    size_t tN = 1; // TODO: TU MUSI WEJSC NR PIERWSZEGO ELEMENTU (O ILE S� W OG�LE NUMEROWANE, JE�LI NIE TO DA� 1)
    size_t tE = 1; // first element

    size_t tSize, tLoLeftNr = 0, tLoRightNr = 0, tUpLeftNr = 0, tUpRightNr = 0, // element nodes numebers
           tFstArg = 0, tSecArg = 0; // assistant values to set vector A
    tSize = tNoOfNodesX+2;

    double tValLoLeft = 0., tValLoRight = 0., tValUpLeft = 0., tValUpRight = 0., // assistant values of nodes parameters
           tKXAssist = 0., tKYAssist = 0., tElemWidth = 0., tElemHeight = 0., tF = 0., // assistant values to set K components
           tK11 = 0., tK21 = 0., tK31 = 0., tK41 = 0., tK22 = 0., tK32 = 0., tK42 = 0., tK33 = 0., tK43 = 0., tK44 = 0.; // local symetric matrix components

    // put zeros to the matrix
    std::fill(ipA->begin(), ipA->end(), 0.);
    std::fill(ipB->begin(), ipB->end(), 0.);

    // set vector A and vector B
    for (size_t i=1; i<=tNoOfElements; ++i)
    {
        // set element nodes numbers
        tLoLeftNr = 1; // TODO: TU MUSI WEJSC NR ODPOWIEDNIEGO W�Z�A
        tLoRightNr = 2; // TODO: TU MUSI WEJSC NR ODPOWIEDNIEGO W�Z�A
        tUpLeftNr = 6; // TODO: TU MUSI WEJSC NR ODPOWIEDNIEGO W�Z�A
        tUpRightNr = 7; // TODO: TU MUSI WEJSC NR ODPOWIEDNIEGO W�Z�A

        // set elements size
        tElemWidth = 10.; // TODO: TU TRZEBA WSTAWIC SZEROKOSC ELEMENTU
        tElemHeight = 5.; // TODO: TU TRZEBA WSTAWIC WYSOKOSC ELEMENTU

        tKXAssist = 44.; // TODO: TU TRZEBA WSTAWIC PRZEWODNOSC CIEPLNA W KIERUNKU X-OWYM
        tKYAssist = 22.; // TODO: TU TRZEBA WSTAWIC PRZEWODNOSC CIEPLNA W KIERUNKU Y-OWYM

        // load vector
        double tHeat = 0.; // heat sources

        tF = 0.25*tElemWidth*tElemHeight*(tHeat); // TODO: DODA� �R�D�A

        // calculating K
        tK22 = tK11 = 0.5 * tKYAssist;
        tK21 = -tK11;
        tK44 = tK33 = tK22 = tK11 = (tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight) / 3.;
        tK43 = tK21 = (-2.*tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight ) /6.;
        tK42 = tK31 = -(tKXAssist*tElemHeight/tElemWidth + tKYAssist*tElemWidth/tElemHeight)/6.;
        tK32 = tK41 = (tKXAssist*tElemHeight/tElemWidth -2.*tKYAssist*tElemWidth/tElemHeight)/6.;

        // set Matrix A
        ipA->at((tLoLeftNr-1)*tSize) +=  tK11;
        ipA->at((tLoRightNr-1)*tSize) +=  tK22;
        ipA->at((tUpRightNr-1)*tSize) += tK33;
        ipA->at((tUpLeftNr-1)*tSize) += tK44;

        //tLoRightNr > tLoLeftNr ? (tFstArg = tLoRightNr, tSecArg = tLoLeftNr) : (tFstArg = tLoLeftNr, tSecArg = tLoRightNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK21 + tG21); // 1D2D
        ipE->at(tLoNr-1) += (tK21 + tG21); // 1D2D

        //tUpRightNr > tLoLeftNr ? (tFstArg = tUpRightNr, tSecArg = tLoLeftNr) : (tFstArg = tLoLeftNr, tSecArg = tUpRightNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK31 + tG31); // 1D2D

        //tUpLeftNr > tLoLeftNr ? (tFstArg = tUpLeftNr, tSecArg = tLoLeftNr) : (tFstArg = tLoLeftNr, tSecArg = tUpLeftNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK41 + tG41); // 1D2D

        //tUpRightNr > tLoRightNr ? (tFstArg = tUpRightNr, tSecArg = tLoRightNr) : (tFstArg = tLoRightNr, tSecArg = tUpRightNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK32 + tG32); // 1D2D

        //tUpLeftNr > tLoRightNr ? (tFstArg = tUpLeftNr, tSecArg = tLoRightNr) : (tFstArg = tLoRightNr, tSecArg = tUpLeftNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK42 + tG42); // 1D2D

        //tUpLeftNr > tUpRightNr ? (tFstArg = tUpLeftNr, tSecArg = tUpRightNr) : (tFstArg = tUpRightNr, tSecArg = tUpLeftNr); // 1D2D
        //ipA->at((tSecArg-1)*tSize+tFstArg-tSecArg) += (tK43 + tG43); // 1D2D

        // get element nodes values
        tValLoLeft = 300.; // TODO: TU TRZEBA WSTAWIC TEMPERATUR� W TYM WʏLE
        tValLoRight = 300.; // TODO: TU TRZEBA WSTAWIC TEMPERATUR� W TYM WʏLE
        tValUpLeft = 300.; // TODO: TU TRZEBA WSTAWIC TEMPERATUR� W TYM WʏLE
        tValUpRight = 300.; // TODO: TU TRZEBA WSTAWIC TEMPERATUR� W TYM WʏLE

        // set vector B
        ipB->at(tLoLeftNr-1)  += -(tK11*tValLoLeft + tK21*tValLoRight + tK31*tValUpRight + tK41*tValUpLeft) + tF;
        ipB->at(tLoRightNr-1) += -(tK21*tValLoLeft + tK22*tValLoRight + tK32*tValUpRight + tK42*tValUpLeft) + tF;
        ipB->at(tUpRightNr-1) += -(tK31*tValLoLeft + tK32*tValLoRight + tK33*tValUpRight + tK43*tValUpLeft) + tF;
        ipB->at(tUpLeftNr-1)  += -(tK41*tValLoLeft + tK42*tValLoRight + tK43*tValUpRight + tK44*tValUpLeft) + tF;

    }
    // Add Big Number to Data A (//if (ttN->getVolContSide() != 'M' || ttN->ifTConst()))
    for (ttN = mNodes.begin(); ttN != mNodes.end(); ++ttN)
        if ((iType == "T" && ttN->ifTConst()) || (iType != "T" && ttN->getVolContSide() != 'M'))
            ipD->at(ttN->getNr()-1) += cRun::BigNum; // 1D2D
            //ipA->at(tSize*(ttN->getNr()-1)) += cRun::BigNum; // 1D2D

*/
}

void FiniteElementMethodThermal2DSolver::findNewVectorOfTemp()
{
}

void FiniteElementMethodThermal2DSolver::calculateT()
{
}

}}} // namespace plask::solvers::finiteT
