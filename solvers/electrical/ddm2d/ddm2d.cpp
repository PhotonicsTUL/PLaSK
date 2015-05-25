#include "ddm2d.h"

namespace plask { namespace solvers { namespace electrical {

template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::DriftDiffusionModel2DSolver(const std::string& name) :
    SolverWithMesh<Geometry2DType, RectangularMesh<2>>(name),
    //pcond(5.),
    //ncond(50.),
    //loopno(0),
    //default_junction_conductivity(5.),
    //maxerr(0.05),
    //heatmet(HEAT_JOULES),
//     outPotential(this, &DriftDiffusionModel2DSolver<Geometry2DType>::getPotentials),
    //outCurrentDensity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getCurrentDensities),
    //outHeat(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getHeatDensities),
    //outConductivity(this, &FiniteElementMethodElectrical2DSolver<Geometry2DType>::getConductivity),
    //algorithm(ALGORITHM_CHOLESKY),
    //itererr(1e-8),
    //iterlim(10000),
    //logfreq(500),
    mRsrh(false),
    mRrad(false),
    mRaug(false),
    mPol(false),
    mFullIon(true)
{
    onInvalidate();
    inTemperature = 300.;
}

template<typename Geometry2DType>
DriftDiffusionModel2DSolver<Geometry2DType>::~DriftDiffusionModel2DSolver() {
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::loadFile(std::string filename)
{
    fnStructure = filename;
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::compute()
{
    writelog(LOG_INFO, "Begining DDM calculations");

    loadFile("str.txt");

    this->initCalculation(); // This must be called before any calculation!

    mFnFpCalc = "many";
    setStructure();
    int zzz;
    std::cin >> zzz;
    saveResP("results/res_polarizations.txt");
    setScaleParam();
    setMeshPoints();
    setNodes();
    setElements();
    calcPsiI();
    int s = solve();
    if (s<0) std::cout << "ERROR\n";

    return 0.;

    writelog(LOG_RESULT, "Found new values of potential and quasi-Fermi levels");
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::setStructure()
{
    int tNo;
	double tThicknessX, tThicknessY, tX, tY, tZ, tDopConc;
	std::string tMaterial, tDopant, tID;

    std::ifstream f;
    f.open(fnStructure.c_str());

    f >> nl;
    std::cout << "Number of layers in the structure: " << nl << "\n";

    for (int i=0; i<nl; ++i) {
        f >> tNo >> tThicknessX >> tThicknessY >> tMaterial >> tX >> tY >> tZ >> tDopant >> tDopConc >> tID;
        Layer2D layer(tNo, tThicknessX, tThicknessY, tMaterial, tX, tY, tZ, tDopant, tDopConc, tID);
        vL.push_back(layer);
        layer.getInfo();
    }
    //int zzz;
    //std::cin >> zzz;
    f.close();
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::setScaleParam()
{
    scaleT = 300.;
    scaleE = phys::kB_eV*scaleT;
    scaleN = 1e18;
    scaleEpsR = 10.;
    scaleX = sqrt((phys::epsilon0*phys::kB_J*scaleT*scaleEpsR)/(phys::qe*phys::qe*scaleN))*1e3;
    scaleK = 100.;
    scaleMi = 1000.;
    scaleR = ((phys::kB_J*scaleT*scaleMi*scaleN)/(phys::qe*scaleX*scaleX))*1e8;
    scaleJ = ((phys::kB_J*scaleN)*scaleT*scaleMi/scaleX)*10.;
    scalet = scaleN/scaleR;
    scaleB = scaleR/(scaleN*scaleN);
    scaleC = scaleR/(scaleN*scaleN*scaleN);
    scaleH = ((scaleK*scaleT)/(scaleX*scaleX))*1e12;

    /*
    double tT = 0.,
        tx = 0.,
        tEpsR = 0.,
        tN = 0.,
        tMi = 0.,
        //tH = 0.,
        tJ = 0.,
        tR = 0.,
        tK = 0.;

    for (std::vector<Layer2D>::iterator it=vL.begin(); it!=vL.end(); ++it) {
        if (it->getEpsR() > tEpsR) tEpsR = it->getEpsR();
        if (it->getDopConc() > tN) tN = it->getDopConc();
        if (it->getMiN() > tMi) tMi = it->getMiN();
        if (it->getMiP() > tMi) tMi = it->getMiP();
        if (it->getK() > tK) tK = it->getK();
    }
    tT = 300.;
    tx = sqrt((phys::epsilon0*phys::kB_J*300.*tEpsR)/(phys::qe*phys::qe*tN))*1e3, // sometimes detoned as LD (um)
    tJ = ((phys::kB_J*tN)*tT*tMi/tx)*10.,
    tR = ((phys::kB_J*tT*tMi)/(phys::qe*tN*tx*tx))*1e8;
    std::cout << "Scalling parameters:\n";
    std::cout << "T: " << tT << "\n";
    std::cout << "Ld: " << tx << "\n";
    std::cout << "EpsRmax: " << tEpsR << "\n";
    std::cout << "Nmax: " << tN << "\n";
    std::cout << "Mimax: " << tMi << "\n";
    std::cout << "Jx: " << tJ << "\n";
    std::cout << "Rx: " << tR << "\n";
    std::cout << "K: " << tK << "\n";*/
    //int ccc;
    //std::cin >> ccc;
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::setMeshPoints()
{
    nx = this->mesh->axis0->size();
    std::cout << "Number of x-nodes: " << nx << std::endl;
    vX.clear();
    for (int i=0; i<nx; ++i) {
        //std::cout << this->mesh->axis0->at(i) << "\t"; // TEST
        vX.push_back(this->mesh->axis0->at(i)); // vX stores node x-positions (um)
    }
    std::cout << "\n";

    ny = this->mesh->axis1->size();
    std::cout << "Number of y-nodes: " << ny << std::endl;
    vY.clear();
    for (int i=0; i<ny; ++i) {
        //std::cout << this->mesh->axis1->at(i) << "\t"; // TEST
        vY.push_back(this->mesh->axis1->at(i)); // vY stores node y-positions (um)
    }
    std::cout << "\n";

    std::cout << "Mesh->size: " << this->mesh->size() << "\n"; // TEST

    // Set stiffness matrix and load vector // TEST
    /*for (auto e: this->mesh->elements) {
        size_t i = e.getIndex();
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();
        std::cout << i << " " << loleftno << " " << lorghtno << " " << upleftno << " " << uprghtno << " " << e.getLower0() << " " << e.getUpper0() << " " << e.getLower1() << " " << e.getUpper1() << "\n";
    }*/

    int zzz;
    std::cin >> zzz;

    return 0;
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::setNodes()
{
    std::cout << "Setting nodes..\n";
    nn = nx*ny;
    std::cout << "Number of nodes in the structure: " << nn << "\n";
    vN.clear();
    for (int j=0; j<ny; ++j) {
        for (int i=0; i<nx; ++i) {
            Node2D node(j*nx+i+1, (vX[i])/scaleX, (vY[j])/scaleX, checkPos(vY[j])); // TODO13
            vN.push_back(node);
        }
    }

    // boundary conditions for temperature
    /*for (int i=0; i<nn; ++i) {
        if (vN[i].getSide() == 'p') vN[i].setT((300.)/scaleT);
        else if (vN[i].getSide() == 'n') vN[i].setT((300.)/scaleT);
        else vN[i].setT((300.)/scaleT);
    }*/

    /*int zzz;
    std::cin >> zzz;*/

    return 0;
}

template<typename Geometry2DType>
char DriftDiffusionModel2DSolver<Geometry2DType>::checkPos(double iY)
{
    if (iY == 0.) return 'p';

    double tY(0.);
    for (std::vector<Layer2D>::iterator it=vL.begin(); it!=vL.end(); ++it) {
        tY += it->getThicknessY();
        if ((areEq(iY, tY, dyAcc)) && (it==vL.end()-1))
            return 'n';
        else if (areEq(iY, tY, dyAcc))
            return 'i';
    }
    return '-';
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::setElements()
{
    std::cout << "Setting elements..\n";
    ne = (nx-1)*(ny-1);//nn-1;
    std::cout << "Number of elements in the structure: " << ne << "\n";
    vE.clear();

    for (int j=0; j<ny-1; ++j) {
        for (int i=0; i<nx-1; ++i) {
            Layer2D* tL = checkLay((0.5*(vN[j*nx+i].getY()+vN[(j+1)*nx+i].getY()))*scaleX);
            if (tL == NULL)
                return -1;
            else {
                Elem2D elem(j*(nx-1)+i+1, &vN[j*nx+i], &vN[j*nx+i+1], &vN[(j+1)*nx+i+1], &vN[(j+1)*nx+i], tL);
                vE.push_back(elem);
            }
        }
    }

    /*for (int i=0; i<ne; ++i) {
        vE[i].getInfo();
    }
    int zzz;
    std::cin >> zzz;*/

    return 0;
}

template<typename Geometry2DType>
Layer2D* DriftDiffusionModel2DSolver<Geometry2DType>::checkLay(double iY)
{
    double tY(0.);
    for (std::vector<Layer2D>::iterator it=vL.begin(); it!=vL.end(); ++it) {
        if ((iY>tY) && (iY<tY+it->getThicknessY()))
            return &(*it);
        else
            tY += it->getThicknessY();
    }
    return NULL;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::calcPsiI()
{
    std::cout << "Calculating initial potential..\n";
    double Psi0(0.);
    std::string prevMat = " "; // previous material
    for (int i=0; i<ne; ++i) {
        //double x = vE[i].getX(); // normalised x (-)
        if (prevMat != vE[i].getL()->getID()) {
            Psi0 = findPsiI(vE[i].getL()->getEc0Norm(),vE[i].getL()->getEv0Norm(),vE[i].getL()->getNcNorm(),vE[i].getL()->getNvNorm(),vE[i].getL()->getNdNorm(),vE[i].getL()->getNaNorm(),vE[i].getL()->getEdNorm(),vE[i].getL()->getEaNorm(),exp(0.),exp(0.),300.);
            prevMat = vE[i].getL()->getID();
        }
        vE[i].getN1Ptr()->setPsi( (vE[i].getN1Ptr()->getFlagPsi0() * vE[i].getN1Ptr()->getPsi() + Psi0) / (vE[i].getN1Ptr()->getFlagPsi0() + 1.) );
        vE[i].getN1Ptr()->setFlagPsi0(vE[i].getN1Ptr()->getFlagPsi0()+1);
        vE[i].getN2Ptr()->setPsi( (vE[i].getN2Ptr()->getFlagPsi0() * vE[i].getN2Ptr()->getPsi() + Psi0) / (vE[i].getN2Ptr()->getFlagPsi0() + 1.) );
        vE[i].getN2Ptr()->setFlagPsi0(vE[i].getN2Ptr()->getFlagPsi0()+1);
        vE[i].getN3Ptr()->setPsi( (vE[i].getN3Ptr()->getFlagPsi0() * vE[i].getN3Ptr()->getPsi() + Psi0) / (vE[i].getN3Ptr()->getFlagPsi0() + 1.) );
        vE[i].getN3Ptr()->setFlagPsi0(vE[i].getN3Ptr()->getFlagPsi0()+1);
        vE[i].getN4Ptr()->setPsi( (vE[i].getN4Ptr()->getFlagPsi0() * vE[i].getN4Ptr()->getPsi() + Psi0) / (vE[i].getN4Ptr()->getFlagPsi0() + 1.) );
        vE[i].getN4Ptr()->setFlagPsi0(vE[i].getN4Ptr()->getFlagPsi0()+1);

        if (!i) {
            PsiP = (Psi0)*scaleE;
            std::cout << "Build in potential: " << PsiP << "\n";
        }

    }
    for (int i=0; i<ne; ++i) {
        vE[i].setN(calcN(vE[i].getL()->getNcNorm(), vE[i].getFnEta(), vE[i].getPsi(), vE[i].getL()->getEc0Norm(), vE[i].getT()));
        vE[i].setP(calcP(vE[i].getL()->getNvNorm(), vE[i].getFpKsi(), vE[i].getPsi(), vE[i].getL()->getEv0Norm(), vE[i].getT()));
        vE[i].setN1(calcN(vE[i].getL()->getNcNorm(), vE[i].getN1Ptr()->getFnEta(), vE[i].getPsi(), vE[i].getL()->getEc0Norm(), vE[i].getN1Ptr()->getT()));
        vE[i].setN2(calcN(vE[i].getL()->getNcNorm(), vE[i].getN2Ptr()->getFnEta(), vE[i].getPsi(), vE[i].getL()->getEc0Norm(), vE[i].getN2Ptr()->getT()));
        vE[i].setN3(calcN(vE[i].getL()->getNcNorm(), vE[i].getN3Ptr()->getFnEta(), vE[i].getPsi(), vE[i].getL()->getEc0Norm(), vE[i].getN3Ptr()->getT()));
        vE[i].setN4(calcN(vE[i].getL()->getNcNorm(), vE[i].getN4Ptr()->getFnEta(), vE[i].getPsi(), vE[i].getL()->getEc0Norm(), vE[i].getN4Ptr()->getT()));
        vE[i].setP1(calcP(vE[i].getL()->getNvNorm(), vE[i].getN1Ptr()->getFpKsi(), vE[i].getPsi(), vE[i].getL()->getEv0Norm(), vE[i].getN1Ptr()->getT()));
        vE[i].setP2(calcP(vE[i].getL()->getNvNorm(), vE[i].getN2Ptr()->getFpKsi(), vE[i].getPsi(), vE[i].getL()->getEv0Norm(), vE[i].getN2Ptr()->getT()));
        vE[i].setP3(calcP(vE[i].getL()->getNvNorm(), vE[i].getN3Ptr()->getFpKsi(), vE[i].getPsi(), vE[i].getL()->getEv0Norm(), vE[i].getN3Ptr()->getT()));
        vE[i].setP4(calcP(vE[i].getL()->getNvNorm(), vE[i].getN4Ptr()->getFpKsi(), vE[i].getPsi(), vE[i].getL()->getEv0Norm(), vE[i].getN4Ptr()->getT()));
    }

    // below code integrate with PLaSK

    std::cout << "INTEGRATE WITH PLASK\n";

    //double oldEg(-1.), oldNd(-1.), oldNa(-1); // old Eg, Nd, Na values
    std::string oldID("-"), tmpID; // old material ID / temporary material ID
    //double tmpEg, tmpNd, tmpNa; // temporary Eg, Nd, Na values
    double oldPsiI(-1.), tmpPsiI; // initial potential


    for (auto e: this->mesh->elements) {
        size_t i = e.getIndex();
        size_t loleftno = e.getLoLoIndex();
        size_t lorghtno = e.getUpLoIndex();
        size_t upleftno = e.getLoUpIndex();
        size_t uprghtno = e.getUpUpIndex();
        Vec<2,double> midpoint = e.getMidpoint();
        //tmpEg = this->geometry->getMaterial(midpoint)->Eg(300.);
        tmpID = this->geometry->getMaterial(midpoint)->name();
        if (tmpID!=oldID)
        {
            tmpPsiI = findPsiI(this->geometry->getMaterial(midpoint)->CB(300.,0.,'G')/scaleE, this->geometry->getMaterial(midpoint)->VB(300.,0.,'G')/scaleE,
                               this->geometry->getMaterial(midpoint)->Nc(300.,0.,'G')/scaleN, this->geometry->getMaterial(midpoint)->Nv(300.,0.,'G')/scaleN,
                               this->geometry->getMaterial(midpoint)->Nd()/scaleN, this->geometry->getMaterial(midpoint)->Na()/scaleN,
                               this->geometry->getMaterial(midpoint)->EactD(300.)/scaleE, this->geometry->getMaterial(midpoint)->EactA(300.)/scaleE,
                               exp(0.),exp(0.),300.);
            oldID = tmpID;
            int ccc;
            std::cin >> ccc;
        }


        std::cout << i << " " << /*tmpEg <<*/ " " << tmpID << "\n";// << upleftno << " " << uprghtno << " " << e.getLower0() << " " << e.getUpper0() << " " << e.getLower1() << " " << e.getUpper1() << "\n";
    }

    //int zzz;
    //std::cin >> zzz;
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::solve()
{
    std::vector<double> mpAA(vN.size()*(nx+2), 0.);
	std::vector<double> mpDD(vN.size(), 0.);
	std::vector<double> mpYY(vN.size(), 0.);

	std::cout << "Solving with the use of FEM.." << std::endl;
	try
	{
        if (1) { // save initial values
            std::stringstream s1("");
            std::stringstream s2("");
            s1 << "results/res_nodes(initial).txt";
            s2 << "results/res_elements(initial).txt";
            saveResN(s1.str());
            saveResE(s2.str());
        }

        double U = 0.; // current voltage
		///Calculations status (0 - interrupted/default, 1 - correction limited, -1 - loop limited
		int statT = 0, statPsi = 0, statFn = 0, statFp = 0;

		std::cout << "Starting calculations..\n";

		// --------------- U = 0 V (thermal model only) ---------------------------------------
        /*if (1) { // TODO13
            if (!U) {
                std::cout << "U = 0 V.";
                std::cout << "\nCalculating T.. ";
                statT = solvePart(mpAA, mpDD, mpYY, "T", loopT, (accT)/scaleT);
                if (statT<0) return -1;
            }
            std::stringstream s1("");
            s1 << "results/res_nodesT(" << U << "V).txt";
            saveResN(s1.str());
        }*/

		// --------------- U = 0 V ------------------------------------------------------------
		if (1) {
		    if (!U) {
                std::cout << "\nU = 0 V.";
                std::cout << "\nCalculating E.. ";
                statPsi = solvePart(mpAA, mpDD, mpYY, "Psi0", loopPsi0, (accPsi0)/scaleE);
                if (statPsi<0) return -1;
            }
            std::stringstream s1("");
            std::stringstream s2("");
            s1 << "results/res_nodes(" << U << "V).txt";
            s2 << "results/res_elements(" << U << "V).txt";
            saveResN(s1.str());
            saveResE(s2.str());
            saveResJ(U);
		}

		// --------------- U > 0 V ------------------------------------------------------------
        if (1) {
            int nU = static_cast<int>(round(Umax/dU));
            std::cout << "\nvoltage steps: " << nU;
            U += dU;
            //std::cout << "Ppsi " << func::rescaleE(vN[0].getPsi()) << "Npsi " << func::rescaleE(vN[vN.size()-1].getPsi()) << "\n";
            for (int i=1; i<=nU; ++i) {
                std::cout << "\nU = " << U << " V.";
                updVoltNodes(U);
                //Voltage Step (Psi-Fn-Fp) Loop Calculations
                int loop = 0;
                while (loop<loopPsiFnFp) {
                    std::cout << "\nCalculating E.. ";
                    statPsi = solvePart(mpAA, mpDD, mpYY, "Psi", loopPsi, (accPsi)/scaleE);
                    if (statPsi<0) return -1;
                    std::cout << "\nCalculating Fn.. ";
                    statFn = solvePart(mpAA, mpDD, mpYY, "Fn", loopFnFp, (accFnFp)/scaleE);
                    if (statFn<0) return -1;
                    std::cout << "\nCalculating Fp.. ";
                    statFp = solvePart(mpAA, mpDD, mpYY, "Fp", loopFnFp, (accFnFp)/scaleE);
                    if (statFp<0) return -1;

                    ++loop;

                    if ((statPsi==1)&&(statFn==1)&&(statFp==1)&&(loopPsi!=1)&&(loopFnFp!=1)) break; // accuracy achieved in one loop for all variables
                }

                // saving results to files
                std::stringstream s1("");
                std::stringstream s2("");
                s1 << "results/res_nodes(" << U << "V).txt";
                s2 << "results/res_elements(" << U << "V).txt";
                if (!(i%dUsav)) {
                    saveResN(s1.str());
                    saveResE(s2.str());
                }
                saveResJ(U); // 4 values per one U step - can be written to file as often as possible
                U += dU;
            }
        }
		return 0;
	}
	catch(...)
	{
		return -1;
	}
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::solvePart(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, const std::string& iType, int iLoopLim, double iCorrLim) // 1D2D
{
    std::vector<double>::const_iterator ttCorrPos, ttCorrNeg;
    int tLoop = iLoopLim, tInfo = 1;//, tSize = 0;
	double tMaxRelUpd = 1e20;

    while (tLoop && tMaxRelUpd > iCorrLim) {
		setSolverData(ipAA, ipDD, ipYY, iType);
		tInfo = solveEquationAYD(ipAA, ipDD, ipYY, nn);
		if (tInfo) return -1;

		if (!tInfo) {
			tMaxRelUpd = updNodes(ipYY, iType); // uUpdate nodes and get max. correction
			updElements();

			//Find Max Correction
			/*ttCorrPos = std::max_element(ipY->begin(), ipY->end());
			ttCorrNeg = std::min_element(ipY->begin(), ipY->end());
			fabs(*ttCorrNeg) > *ttCorrPos ? tCorr = std::abs(*ttCorrNeg) : tCorr = *ttCorrPos;
			//Show Max Correction
			if (iType == "T")
                std::cout << func::rescaleT(tCorr) << " ";
			else if ((iType == "Psi") || (iType == "Psi0") || (iType == "Fn") || (iType == "Fp"))
                std::cout << func::rescaleE(tCorr) << " ";*/

            std::cout << tMaxRelUpd << " ";
			--tLoop;
		}
		else if (tInfo < 0) return -1;
		else return -2;
    }

	if (tLoop == iLoopLim-1) return 1; // one loop was enough to achieve accuracy
	else if (tMaxRelUpd <= iCorrLim-1) return 2; // accuracy achieved in more than one loop
	else if (tLoop < iLoopLim-1) return 3; // loop limit achieved
	else return -1;
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::solveEquationAYD(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, int n)
{
    int tRes = SolveEquation(&(*ipAA.begin()), &(*ipDD.begin()), n, nx+1);

    if (tRes) {
        std::cout << "Calculation error\n";
        return -1;
    }

    for (int i=0; i<n; ++i)
        ipYY.at(i) = ipDD.at(i);

    return tRes;
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::setSolverData(std::vector<double> &ipAA, std::vector<double> &ipDD, std::vector<double> &ipYY, const std::string& iType)
{
	std::vector<Elem2D>::iterator ttE = vE.begin();
    std::vector<Node2D>::iterator ttN = vN.begin();

    size_t tLoLeNo(0), tLoRiNo(0), tUpLeNo(0), tUpRiNo(0);

    double tVal1(0.), tVal2(0.), tVal3(0.), tVal4(0.), // old parameter values in nodes
           tKtmp(0.), tKtmpX(0.), tKtmpY(0.), tGtmp(0.), tFtmp(0.), // tmp values to set K and G
           tK11(0.), tK21(0.), tK31(0.), tK41(0.), tK22(0.), tK32(0.), tK42(0.), tK33(0.), tK43(0.), tK44(0.),
           tG11(0.), tG21(0.), tG31(0.), tG41(0.), tG22(0.), tG32(0.), tG42(0.), tG33(0.), tG43(0.), tG44(0.),
           tF1(0.), tF2(0.), tF3(0.), tF4(0.),
           hx(0.), hy(0.), n(0.), p(0.), ni(0.), Ne(0.), Nh(0.);

    // set vectors (zeros)
    std::fill(ipAA.begin(), ipAA.end(), 0.);
    std::fill(ipDD.begin(), ipDD.end(), 0.);
    std::fill(ipYY.begin(), ipYY.end(), 0.);

    // set system of equations
	for (ttE = vE.begin(); ttE != vE.end(); ++ttE) {
        tLoLeNo = ttE->getN1Ptr()->getNo();
        tLoRiNo = ttE->getN2Ptr()->getNo();
        tUpRiNo = ttE->getN3Ptr()->getNo();
        tUpLeNo = ttE->getN4Ptr()->getNo();

        hx = ttE->getSizeX();
        hy = ttE->getSizeY();
        n = ttE->getN();
        p = ttE->getP();
        Ne = ttE->getL()->getNcNorm() * exp(ttE->getPsi()-ttE->getL()->getEc0Norm());
        Nh = ttE->getL()->getNvNorm() * exp(-ttE->getPsi()+ttE->getL()->getEv0Norm());
        ni = ttE->getL()->getNiNorm();

        double yn(0.);
        if (stat == "MB") yn = 1.;
        else if (stat == "FD") yn = calcFD12(log(ttE->getFnEta())+ttE->getPsi()-ttE->getL()->getEc0Norm())/(ttE->getFnEta()*exp(ttE->getPsi()-ttE->getL()->getEc0Norm()));

        double yp(0.);
        if (stat == "MB") yp = 1.;
        else if (stat == "FD") yp = calcFD12(log(ttE->getFpKsi())-ttE->getPsi()+ttE->getL()->getEv0Norm())/(ttE->getFpKsi()*exp(-ttE->getPsi()+ttE->getL()->getEv0Norm()));

        // set some helpful parameters
        if (iType == "T") {
			tKtmp = 1. / (3.*(hx*0.5)*(hy*0.5));
            tKtmpX = ttE->getL()->getKNorm() * (hy*0.5) * (hy*0.5);
            tKtmpY = ttE->getL()->getKNorm() * (hx*0.5) * (hx*0.5);
            tGtmp = 0.;
            tFtmp = 0.;
		}
        else if ((iType == "Psi") || (iType == "Psi0")) {
			tKtmp = 1. / (3.*(hx*0.5)*(hy*0.5));
            tKtmpX = ttE->getL()->getEpsRNorm() * (hy*0.5) * (hy*0.5);
            tKtmpY = ttE->getL()->getEpsRNorm() * (hx*0.5) * (hx*0.5);
            tGtmp = (1./9.) * (p + n) * (hx*0.5) * (hy*0.5);
            double iNdIon = ttE->getL()->getNdNorm();
            double iNaIon = ttE->getL()->getNaNorm();
            if (!mFullIon)
            {
                double gD(2.), gA(4.);
                double iNdTmp = (ttE->getL()->getNcNorm()/gD)*exp(-ttE->getL()->getEdNorm());
                double iNaTmp = (ttE->getL()->getNvNorm()/gA)*exp(-ttE->getL()->getEaNorm());
                iNdIon = ttE->getL()->getNdNorm() * (iNdTmp/(iNdTmp+n));
                iNaIon = ttE->getL()->getNaNorm() * (iNaTmp/(iNaTmp+p));
            }
			tFtmp = - (hx*0.5) * (hy*0.5) * (p - n + iNdIon - iNaIon);
		}
		else if (iType == "Fn") {
			tKtmp = 1. / (3.*(hx*0.5)*(hy*0.5));
            tKtmpX = ttE->getL()->getMiNNorm() * Ne * yn * (hy*0.5) * (hy*0.5);
            tKtmpY = ttE->getL()->getMiNNorm() * Ne * yn * (hx*0.5) * (hx*0.5);
            tFtmp = tGtmp = 0.;

            if (ttE->getL()->getID()=="active") {
                if (mRsrh) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Ne * yn * (p + ni) * (ttE->getL()->getTpNorm() * ni + ttE->getL()->getTnNorm() * p)
                        / pow(ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni), 2.));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (n * p - ni * ni) / (ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni)));
                }
                if (mRrad) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * Ne * yn * p);
                    tFtmp += ((hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * (n * p - ni * ni));
                }
                if (mRaug) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Ne * yn * ((ttE->getL()->getCnNorm() * (2. * n * p - ni * ni) + ttE->getL()->getCpNorm() * p * p)));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (ttE->getL()->getCnNorm() * n + ttE->getL()->getCpNorm() * p) * (n * p - ni * ni));
                }
			}
		}
		else if (iType == "Fp")	{
			tKtmp = 1. / (3.*(hx*0.5)*(hy*0.5));
            tKtmpX = ttE->getL()->getMiPNorm() * Nh * yp * (hy*0.5) * (hy*0.5);
            tKtmpY = ttE->getL()->getMiPNorm() * Nh * yp * (hx*0.5) * (hx*0.5);
            tFtmp = tGtmp = 0.;

            if (ttE->getL()->getID()=="active") { // TODO (only in active?)
                if (mRsrh) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Nh * yp * (n + ni) * (ttE->getL()->getTnNorm() * ni + ttE->getL()->getTpNorm() * n)
                        / pow(ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni), 2.));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (n * p - ni * ni) / (ttE->getL()->getTpNorm() * (n + ni) + ttE->getL()->getTnNorm() * (p + ni)));
                }
                if (mRrad) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * Nh * yp * n);
                    tFtmp += ((hx*0.5) * (hy*0.5) * ttE->getL()->getBNorm() * (n * p - ni * ni));
                }
                if (mRaug) {
                    tGtmp += ((1./9.) * (hx*0.5) * (hy*0.5) * Nh * yp * ((ttE->getL()->getCpNorm() * (2. * n * p - ni * ni) + ttE->getL()->getCnNorm() * n * n)));
                    tFtmp += ((hx*0.5) * (hy*0.5) * (ttE->getL()->getCnNorm() * n + ttE->getL()->getCpNorm() * p) * (n * p - ni * ni));
                }
			}
		}
        else return -1;

		// local K
        tK11 = tK22 = tK33 = tK44 = (tKtmpX+tKtmpY)*tKtmp;
        tK21 = tK43 = 0.5*(-2.*tKtmpX+tKtmpY)*tKtmp;
        tK31 = tK42 = 0.5*(-tKtmpX-tKtmpY)*tKtmp;
        tK41 = tK32 = 0.5*(tKtmpX-2.*tKtmpY)*tKtmp;
        // local G
        tG11 = tG22 = tG33 = tG44 = 4.*tGtmp;
        tG21 = tG41 = tG32 = tG43 = 2.*tGtmp;
        tG31 = tG42 = 1.*tGtmp;
        // local F
        tF4 = tF3 = tF2 = tF1 = tFtmp;

        // set matrix A
        int tI1(0), tI2(0), tIT(0);
        tI1 = tLoLeNo; tI2 = tLoLeNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK11 + tG11);
        tI1 = tLoLeNo; tI2 = tLoRiNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK21 + tG21);
        tI1 = tLoLeNo; tI2 = tUpRiNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK31 + tG31);
        tI1 = tLoLeNo; tI2 = tUpLeNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK41 + tG41);
        tI1 = tLoRiNo; tI2 = tLoRiNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK22 + tG22);
        tI1 = tLoRiNo; tI2 = tUpRiNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK32 + tG32);
        tI1 = tLoRiNo; tI2 = tUpLeNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK42 + tG42);
        tI1 = tUpRiNo; tI2 = tUpRiNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK33 + tG33);
        tI1 = tUpRiNo; tI2 = tUpLeNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK43 + tG43);
        tI1 = tUpLeNo; tI2 = tUpLeNo; if (tI2 < tI1) { tIT = tI1; tI1 = tI2; tI2 = tIT; } ipAA.at((tI1-1)*(nx+2)+(tI2-tI1)) += (tK44 + tG44);

		// get element nodes values (T,Psi,Fn,Fp)
        if (iType == "T") {
            tVal1 = ttE->getN1Ptr()->getT();
            tVal2 = ttE->getN2Ptr()->getT();
            tVal3 = ttE->getN3Ptr()->getT();
            tVal4 = ttE->getN4Ptr()->getT();
        }
        else if ((iType == "Psi") || (iType == "Psi0")) {
            tVal1 = ttE->getN1Ptr()->getPsi();
            tVal2 = ttE->getN2Ptr()->getPsi();
            tVal3 = ttE->getN3Ptr()->getPsi();
            tVal4 = ttE->getN4Ptr()->getPsi();
        }
        else if (iType == "Fn") {
            tVal1 = ttE->getN1Ptr()->getFnEta();
            tVal2 = ttE->getN2Ptr()->getFnEta();
            tVal3 = ttE->getN3Ptr()->getFnEta();
            tVal4 = ttE->getN4Ptr()->getFnEta();
        }
        else if (iType == "Fp") {
            tVal1 = ttE->getN1Ptr()->getFpKsi();
            tVal2 = ttE->getN2Ptr()->getFpKsi();
            tVal3 = ttE->getN3Ptr()->getFpKsi();
            tVal4 = ttE->getN4Ptr()->getFpKsi();
        }

        // set vector B
		ipDD.at(tLoLeNo-1) += -(tK11*tVal1 + tK21*tVal2 + tK31*tVal3 + tK41*tVal4 + tF1);
		ipDD.at(tLoRiNo-1) += -(tK21*tVal1 + tK22*tVal2 + tK32*tVal3 + tK42*tVal4 + tF2);
		ipDD.at(tUpRiNo-1) += -(tK31*tVal1 + tK32*tVal2 + tK33*tVal3 + tK43*tVal4 + tF3);
		ipDD.at(tUpLeNo-1) += -(tK41*tVal1 + tK42*tVal2 + tK43*tVal3 + tK44*tVal4 + tF4);

        // set zeros
        tLoLeNo = 0; tLoRiNo = 0; tUpLeNo = 0; tUpRiNo = 0;
        tK11 = tK21 = tK31 = tK41 = tK22 = tK32 = tK42 = tK33 = tK43 = tK44 = 0.;
        tG11 = tG21 = tG31 = tG41 = tG22 = tG32 = tG42 = tG33 = tG43 = tG44 = 0.;
        tF1 = tF2 = tF3 = tF4 = 0.;

        /* DO NOT DELETE THIS!
        if ((iType == "Psi") && (mPol) && (ttE->getL()->getThicknessY() <= 0.020)) {
            double eII = (3.188 - ttE->getL()->getAlc()) / ttE->getL()->getAlc(); // TODO
            double eL = -2. * eII * ttE->getL()->getC13() / ttE->getL()->getC33(); // TODO
            double Ppz = ttE->getL()->gete33() * eL + 2. * ttE->getL()->gete13() * eII; // TODO
            double tP = ttE->getL()->getPsp() + Ppz;
            //std::cout << tP << "\n";
            //int ccc;
            //std::cin >> ccc;
            if (ttE->getN1Ptr()->getSide() != '-')
                tF1 -= func::scaleP(tP); // -= wg wzorow
            if (ttE->getN2Ptr()->getSide() != '-')
                tF2 += func::scaleP(tP); // += wg wzorow
             //TODO13 (to trzeba odkomentowac)
        }

        if ((iType == "Psi")||(iType == "Psi0")) { // TODO (zaburza to co jest wyzej)
            if (ttE->getN1Ptr()->getSide() != '-') {
                tG22 = tG11 = tG21 = 0.;
                tF1 = 0.; tF2 = 0.;
            }
            if (ttE->getN2Ptr()->getSide() != '-') {
                tG22 = tG11 = tG21 = 0.;
                tF1 = 0.; tF2 = 0.;
            }
        }*/
	}

    // add big number to matrix A
	for (ttN = vN.begin(); ttN != vN.end(); ++ttN)
		if ((ttN->getSide() == 'p') || (ttN->getSide() == 'n')) {
            ipAA.at((ttN->getNo()-1)*(nx+2)) += bigNum; // only one Kii and fi
        }

    return 0;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::updVoltNodes(double iVolt)
{
	for (std::vector<Node2D>::iterator ttN = vN.begin(); ttN != vN.end(); ++ttN)
        if (ttN->getSide() == 'p') {
            ttN->setPsi((PsiP + iVolt)/scaleE);
            ttN->setFn((-iVolt)/scaleE);
            ttN->setFp((-iVolt)/scaleE);
            ttN->setFnEta(exp(ttN->getFn()));
            ttN->setFpKsi(exp(-ttN->getFp()));
        }
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::updNodes(std::vector<double>& ipRes, const std::string& iType)
{
	std::vector<Node2D>::iterator ttN = vN.begin();
    std::vector<double>::const_iterator ttRes = ipRes.begin();

    double tMaxRelUpd = 0.; // update/old_value = this will be the result

    double mcT = (maxcorrT)/scaleT;
    double mcPsi0 = (maxcorrPsi0)/scaleE;
    double mcPsi = (maxcorrPsi)/scaleE;
    //double mcFnFp = (maxcorrFnFp)/scaleE; // TODO
    while (ttN != vN.end()) {
		if ((ttN->getSide() == '-') || (ttN->getSide() == 'i')) {
            if (iType == "T") {
                double dT = (*ttRes);
                if (dT > mcT) dT = mcT;
                else if (dT < -mcT) dT = -mcT;
                if (std::abs(dT/ttN->getT()) > tMaxRelUpd) tMaxRelUpd = std::abs(dT/ttN->getT());
                ttN->setT(ttN->getT() + dT);
            }
            else if (iType == "Psi") {
                double dPsi = (*ttRes);
                if (dPsi > mcPsi) dPsi = mcPsi;
                else if (dPsi < -mcPsi) dPsi = -mcPsi;
                if (std::abs(dPsi/ttN->getPsi()) > tMaxRelUpd) tMaxRelUpd = std::abs(dPsi/ttN->getPsi());
                ttN->setPsi(ttN->getPsi() + dPsi);
            }
            else if (iType == "Fn") {
                double dFnEta = (*ttRes);
                ttN->setFnEta(ttN->getFnEta() + dFnEta);
                if (std::abs(dFnEta/ttN->getFnEta()) > tMaxRelUpd) tMaxRelUpd = std::abs(dFnEta/ttN->getFnEta());
                if (ttN->getFnEta()>0)
                    ttN->setFn(log(ttN->getFnEta()));
                else
                    ttN->setFn(0.);
            }
            else if (iType == "Fp") {
                double dFpKsi = (*ttRes);
                ttN->setFpKsi(ttN->getFpKsi() + dFpKsi);
                if (std::abs(dFpKsi/ttN->getFpKsi()) > tMaxRelUpd) tMaxRelUpd = std::abs(dFpKsi/ttN->getFpKsi());
                if (ttN->getFpKsi()>0)
                    ttN->setFp(-log(ttN->getFpKsi()));
                else
                    ttN->setFp(0.);
            }
            else if (iType == "Psi0") {
                double dPsi = (*ttRes);
                if (dPsi > mcPsi0) dPsi = mcPsi0;
                else if (dPsi < -mcPsi0) dPsi = -mcPsi0;
                if (std::abs(dPsi/ttN->getPsi()) > tMaxRelUpd) tMaxRelUpd = std::abs(dPsi/ttN->getPsi());
                ttN->setPsi(ttN->getPsi() + dPsi);
            }
        }

        ++ttN;
        ++ttRes;
    }

    return tMaxRelUpd;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::updElements()
{
	for (std::vector<Elem2D>::iterator ttE = vE.begin(); ttE != vE.end(); ++ttE) {
        ttE->setN(calcN(ttE->getL()->getNcNorm(), ttE->getFnEta(), ttE->getPsi(), ttE->getL()->getEc0Norm(), ttE->getT()));
        ttE->setP(calcP(ttE->getL()->getNvNorm(), ttE->getFpKsi(), ttE->getPsi(), ttE->getL()->getEv0Norm(), ttE->getT()));
        ttE->setN1(calcN(ttE->getL()->getNcNorm(), ttE->getN1Ptr()->getFnEta(), ttE->getPsi(), ttE->getL()->getEc0Norm(), ttE->getN1Ptr()->getT()));
        ttE->setN2(calcN(ttE->getL()->getNcNorm(), ttE->getN2Ptr()->getFnEta(), ttE->getPsi(), ttE->getL()->getEc0Norm(), ttE->getN2Ptr()->getT()));
        ttE->setN3(calcN(ttE->getL()->getNcNorm(), ttE->getN3Ptr()->getFnEta(), ttE->getPsi(), ttE->getL()->getEc0Norm(), ttE->getN3Ptr()->getT()));
        ttE->setN4(calcN(ttE->getL()->getNcNorm(), ttE->getN4Ptr()->getFnEta(), ttE->getPsi(), ttE->getL()->getEc0Norm(), ttE->getN4Ptr()->getT()));
        ttE->setP1(calcP(ttE->getL()->getNvNorm(), ttE->getN1Ptr()->getFpKsi(), ttE->getPsi(), ttE->getL()->getEv0Norm(), ttE->getN1Ptr()->getT()));
        ttE->setP2(calcP(ttE->getL()->getNvNorm(), ttE->getN2Ptr()->getFpKsi(), ttE->getPsi(), ttE->getL()->getEv0Norm(), ttE->getN2Ptr()->getT()));
        ttE->setP3(calcP(ttE->getL()->getNvNorm(), ttE->getN3Ptr()->getFpKsi(), ttE->getPsi(), ttE->getL()->getEv0Norm(), ttE->getN3Ptr()->getT()));
        ttE->setP4(calcP(ttE->getL()->getNvNorm(), ttE->getN4Ptr()->getFpKsi(), ttE->getPsi(), ttE->getL()->getEv0Norm(), ttE->getN4Ptr()->getT()));
    }
}

template<typename Geometry2DType>
int DriftDiffusionModel2DSolver<Geometry2DType>::saveResJ(double iU)
{
	double tJny(0.), tJpy(0.), tJy(0.);
	int tN(0);
	for (std::vector<Elem2D>::iterator ttE = vE.begin(); ttE != vE.end(); ++ttE) {
		if ((ttE->getN1Ptr()->getSide() == '-') && (ttE->getN2Ptr()->getSide() == '-'))
        {
            tJny += (ttE->getJny())*scaleJ;
            tJpy += (ttE->getJpy())*scaleJ;
            //tJ += (tJn+tJp);
            tN++;
        }
    }
    tJny = tJny/(1.*tN);
    tJpy = tJpy/(1.*tN);
    tJy = tJny+tJpy;

    std::ofstream tPlik;

    if (!iU)
    {
        tPlik.open ("results/res_currents.txt");
        tPlik << "U\tJny\tJpy\tJy\n";
        tPlik.close();
    }
    tPlik.open ("results/res_currents.txt", std::ios::app);
    //tPlik.seekp(-1,std::ios::end);
    tPlik << iU << "\t" << tJny << "\t" << tJpy << "\t" << tJy << "\n";
    tPlik.close();

    return 0;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveResN(std::string filename)
{
    std::ofstream f;
    f.open(filename.c_str());
    f << "x\ty\tT\tE\tFn\tFp\n";
    for (int i=0; i<nn; ++i)
        if (!(vN[i].getX())*scaleX)
            f << (vN[i].getX())*scaleX << "\t" << (vN[i].getY())*scaleX << "\t" << (vN[i].getT())*scaleT << "\t" << (vN[i].getPsi())*scaleE << "\t" << (log(vN[i].getFnEta()))*scaleE << "\t" << -(log(vN[i].getFpKsi()))*scaleE << "\n";
    f.close();
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveResE(std::string filename)
{
    std::ofstream f;
    f.open(filename.c_str());
    f << "x\ty\tE\tFn\tFp\tEc\tEv\tFy\tN\tP\tJny\tJpy\tRsrh\tRrad\tRaug\tR\n";
    for (int i=0; i<ne; ++i) {
        double Rsrh(0.); if ((mRsrh)/*&&(vE[i].getL()->getID()=="active")*/) Rsrh = (vE[i].getRsrh())*scaleR;
        double Rrad(0.); if ((mRrad)/*&&(vE[i].getL()->getID()=="active")*/) Rrad = (vE[i].getRrad())*scaleR;
        double Raug(0.); if ((mRaug)/*&&(vE[i].getL()->getID()=="active")*/) Raug = (vE[i].getRaug())*scaleR;
        if (!(vE[i].getN1Ptr()->getX())*scaleX)
            f << (vE[i].getX())*scaleX << "\t" << (vE[i].getY())*scaleX << "\t" << (vE[i].getPsi())*scaleE << "\t" << (log(vE[i].getFnEta()))*scaleE << "\t" << -(log(vE[i].getFpKsi()))*scaleE
            << "\t" << vE[i].getL()->getEc0()-(vE[i].getPsi())*scaleE << "\t" << vE[i].getL()->getEv0()-(vE[i].getPsi())*scaleE
            << "\t" << vE[i].getFy()
            << "\t" << (vE[i].getN())*scaleN << "\t" << (vE[i].getP())*scaleN
            << "\t" << (vE[i].getJny())*scaleJ << "\t" << (vE[i].getJpy())*scaleJ
            << "\t" << Rsrh << "\t" << Rrad << "\t" << Raug << "\t" << (Rsrh+Rrad+Raug)
            << "\n";
    }

    f.close();
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::saveResP(std::string filename)
{
    std::ofstream f;
    f.open(filename.c_str());
    f << "ID\tPsp\tPpz\tP\n";
    for (int i=0; i<nl; ++i) {
        double eII = (3.188 - vL[i].getAlc()) / vL[i].getAlc(); // TODO
        double eL = -2. * eII * vL[i].getC13() / vL[i].getC33(); // TODO
        double Ppz = vL[i].gete33() * eL + 2. * vL[i].gete13() * eII; // TODO
        double Ptot = vL[i].getPsp() + Ppz;
        f << vL[i].getID() << "\t" << vL[i].getPsp() << "\t" << Ppz << "\t" << Ptot << "\n";
    }

    f.close();
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::findPsiI(double iEc0, double iEv0, double iNc, double iNv, double iNd, double iNa, double iEd, double iEa, double iFnEta, double iFpKsi, double iT)
{
    double tPsi0(0.), // calculated normalized initial potential
           tPsi0a = (-15.)/scaleE, // normalized edge of the initial range
           tPsi0b = (15.)/scaleE, // normalized edge of the initial range
           tPsi0h = (0.1)/scaleE, // normalized step in the initial range calculations
           tN = 0., tP = 0., // normalized electron/hole concentrations
           tNtot, tNtota = (-1e30)/scaleN, tNtotb = (1e30)/scaleN; // normalized carrier concentration and its initial values for potentials at range edges

    // initial calculations

    int tPsi0n = static_cast<int>(round((tPsi0b-tPsi0a)/tPsi0h)) + 1 ; // number of initial normalized potential values

    std::vector<double> tPsi0v(tPsi0n); // normalized potential values to check
    for (int i=0; i<tPsi0n; ++i)
        tPsi0v[i] = tPsi0a + i*tPsi0h;

    for (int i=0; i<tPsi0n; ++i) {
        tN = calcN(iNc, iFnEta, tPsi0v[i], iEc0, iT);
        tP = calcP(iNv, iFpKsi, tPsi0v[i], iEv0, iT);

        double iNdIon = iNd;
        double iNaIon = iNa;

        if (!mFullIon)
        {
            double gD(2.), gA(4.);
            double iNdTmp = (iNc/gD)*exp(-iEd);
            double iNaTmp = (iNv/gA)*exp(-iEa);
            iNdIon = iNd * (iNdTmp/(iNdTmp+tN));
            iNaIon = iNa * (iNaTmp/(iNaTmp+tP));
        }

        tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

        if (tNtot<0) {
            if (tNtot>tNtota) {
                tNtota = tNtot;
                tPsi0b = tPsi0v[i];
            }
        }
        else if (tNtot>0) {
            if (tNtot<tNtotb) {
                tNtotb = tNtot;
                tPsi0a = tPsi0v[i];
            }
        }
        else // found initial normalised potential
            return tPsi0v[i];
    }

    // precise calculations

    double tPsiUpd = 1e30, // normalised potential update
           tTmpA, tTmpB; // temporary data

    int tL=0; // loop counter
    while ((std::abs(tPsiUpd) > (accPsiI)/scaleE) && (tL < loopPsiI)) {
        tTmpA = (tNtotb-tNtota) / (tPsi0b-tPsi0a);
        tTmpB = tNtota - tTmpA*tPsi0a;
        tPsi0 = - tTmpB/tTmpA; //Psi Check Value
        tN = calcN(iNc, iFnEta, tPsi0, iEc0, iT);
        tP = calcP(iNv, iFpKsi, tPsi0, iEv0, iT);

        double iNdIon = iNd;
        double iNaIon = iNa;

        if (!mFullIon) {
            double gD(2.), gA(4.);
            double iNdTmp = (iNc/gD)*exp(-iEd);
            double iNaTmp = (iNv/gA)*exp(-iEa);
            iNdIon = iNd * (iNdTmp/(iNdTmp+tN));
            iNaIon = iNa * (iNaTmp/(iNaTmp+tP));
        }

        tNtot = tP - tN + iNdIon - iNaIon; // total normalized carrier concentration

        if (tNtot<0) {
            tNtota = tNtot;
            tPsi0b = tPsi0;
        }
        else if (tNtot>0) {
            tNtotb = tNtot;
            tPsi0a = tPsi0;
        }
        else { // found initial normalized potential
            std::cout << "\n" << tL << " loops done. Calculated energy level corresponding to the initial potential: " << (tPsi0)*scaleE << ".\n"; // TEST
            return tPsi0;
        }

        tPsiUpd = tPsi0b-tPsi0a;
        std::cout << (tPsiUpd)*scaleE << " ";
        ++tL;
    }

    std::cout << "\n" << tL << " loops done. Calculated energy level corresponding to the initial potential: " << (tPsi0)*scaleE << ".\n"; // TEST

    return tPsi0;
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcN(double iNc, double iFnEta, double iPsi, double iEc0, double iT)
{
	double yn(0.);
    if (stat == "MB") yn = 1.;
    else if (stat == "FD") yn = calcFD12(log(iFnEta)+iPsi-iEc0)/(iFnEta*exp(iPsi-iEc0));
	return ( iNc*iFnEta*yn*exp(iPsi-iEc0) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcP(double iNv, double iFpKsi, double iPsi, double iEv0, double iT)
{
	double yp(0.);
    if (stat == "MB") yp = 1.;
    else if (stat == "FD") yp = calcFD12(log(iFpKsi)-iPsi+iEv0)/(iFpKsi*exp(-iPsi+iEv0));
	return ( iNv*iFpKsi*yp*exp(iEv0-iPsi) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::calcFD12(double iEta)
{
    double tKsi = pow(iEta,4.) + 33.6*iEta*(1.-0.68*exp(-0.17*(iEta+1.)*(iEta+1.))) + 50.;
    return ( 0.5*sqrt(M_PI) / (0.75*sqrt(M_PI)*pow(tKsi,-0.375)+exp(-iEta)) );
}
/*
template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleX(double iX)
{
    return ( iX / scale::x ); // iX / scaleparamX, where scaleparamX is given in (um)
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleX(double iX)
{
    return ( iX * scale::x ); // iX / scaleparamX, where scaleparamX is given in (um)
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleT(double iT)
{
    return ( iT/scale::T );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleT(double iT)
{
    return ( iT*scale::T );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleE(double iE)
{
    return ( iE/(phys::kB*scale::T) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleE(double iE)
{
    return ( iE*(phys::kB*scale::T) );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::caleN(double iN)
{
    return ( iN/scale::N );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleN(double iN)
{
    return ( iN*scale::N );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleK(double iK)
{
    return ( iK/scale::K );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleEpsR(double iEpsR)
{
    return ( iEpsR/scale::EpsR );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleEpsR(double iEpsR)
{
    return ( iEpsR*scale::EpsR );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleMi(double iMi)
{
    return ( iMi/scale::Mi );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleMi(double iMi)
{
    return ( iMi*scale::Mi );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleJ(double iJ)
{
    return ( iJ*scale::Jx );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleTime(double iTime)
{
    return ( iTime/scale::Time );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleB(double iB)
{
    return ( iB/scale::Bx );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleC(double iC)
{
    return ( iC/scale::Cx );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleH(double iH)
{
    return ( iH/scale::Hx );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::rescaleR(double iR)
{
    return ( iR*scale::Rx );
}

template<typename Geometry2DType>
double DriftDiffusionModel2DSolver<Geometry2DType>::scaleP(double iP)
{
    return ( iP/scale::Px );
}
*/
template<typename Geometry2DType>
bool DriftDiffusionModel2DSolver<Geometry2DType>::areEq(double iVal1, double iVal2, double iTol)
{
    if (std::abs(iVal2-iVal1) <= iTol) return true;
    else return false;
}

template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInitialize()
{
    if (!this->geometry) throw NoGeometryException(this->getId());
    if (!this->mesh) throw NoMeshException(this->getId());
    /*loopno = 0; // TODO skopiowane z femV
    size = this->mesh->size();
    potentials.reset(size, 0.);
    currents.reset(this->mesh->elements.size(), vec(0.,0.));
    conds.reset(this->mesh->elements.size());
    if (junction_conductivity.size() == 1) {
        size_t condsize = 0;
        for (const auto& act: active) condsize += act.right - act.left;
        condsize = max(condsize, size_t(1));
        junction_conductivity.reset(condsize, junction_conductivity[0]);
    }*/
}


template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::onInvalidate() {
    /*conds.reset(); // TODO skopiowane z femV
    potentials.reset();
    currents.reset();
    heats.reset();
    junction_conductivity.reset(1, default_junction_conductivity);*/
}





template<typename Geometry2DType>
void DriftDiffusionModel2DSolver<Geometry2DType>::loadConfiguration(XMLReader &source, Manager &manager)
{
    while (source.requireTagOrEnd())
    {
        std::string param = source.getNodeName();
        // TODO (tylko skopiowane z femV)
        if (param == "voltage" || param == "potential")
            this->readBoundaryConditions(manager, source, voltage_boundary);

        /*else if (param == "loop") {
            maxerr = source.getAttribute<double>("maxerr", maxerr);
            source.requireTagEnd();
        }

        else if (param == "matrix") {
            algorithm = source.enumAttribute<Algorithm>("algorithm")
                .value("cholesky", ALGORITHM_CHOLESKY)
                .value("gauss", ALGORITHM_GAUSS)
                .value("iterative", ALGORITHM_ITERATIVE)
                .get(algorithm);
            itererr = source.getAttribute<double>("itererr", itererr);
            iterlim = source.getAttribute<size_t>("iterlim", iterlim);
            logfreq = source.getAttribute<size_t>("logfreq", logfreq);
            source.requireTagEnd();
        }

        else if (param == "junction") {
            js[0] = source.getAttribute<double>("js", js[0]);
            beta[0] = source.getAttribute<double>("beta", beta[0]);
            auto condjunc = source.getAttribute<double>("pnjcond");
            if (condjunc) setCondJunc(*condjunc);
            auto wavelength = source.getAttribute<double>("wavelength");
            if (wavelength) inWavelength = *wavelength;
            heatmet = source.enumAttribute<HeatMethod>("heat")
                .value("joules", HEAT_JOULES)
                .value("wavelength", HEAT_BANDGAP)
                .get(heatmet);
            for (auto attr: source.getAttributes()) {
                if (attr.first == "beta" || attr.first == "Vt" || attr.first == "js" || attr.first == "pnjcond" || attr.first == "wavelength" || attr.first == "heat") continue;
                if (attr.first.substr(0,4) == "beta") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(4)); }
                    catch (boost::bad_lexical_cast) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setBeta(no, source.requireAttribute<double>(attr.first));
                }
                else if (attr.first.substr(0,2) == "js") {
                    size_t no;
                    try { no = boost::lexical_cast<size_t>(attr.first.substr(2)); }
                    catch (boost::bad_lexical_cast) { throw XMLUnexpectedAttrException(source, attr.first); }
                    setJs(no, source.requireAttribute<double>(attr.first));
                }
                else
                    throw XMLUnexpectedAttrException(source, attr.first);
            }
            source.requireTagEnd();
        }

        else if (param == "contacts") {
            pcond = source.getAttribute<double>("pcond", pcond);
            ncond = source.getAttribute<double>("ncond", ncond);
            source.requireTagEnd();
        }

        else
            this->parseStandardConfiguration(source, manager);*/
    }
}

template<> std::string DriftDiffusionModel2DSolver<Geometry2DCartesian>::getClassName() const { return "electrical.DriftDiffusion2D"; }
template<> std::string DriftDiffusionModel2DSolver<Geometry2DCylindrical>::getClassName() const { return "electrical.DriftDiffusionCyl"; }


template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API DriftDiffusionModel2DSolver<Geometry2DCylindrical>;

}}} // namespace
