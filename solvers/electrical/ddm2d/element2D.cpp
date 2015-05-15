#include "element2D.h"
//------------------------------------------------------------------------------
Elem2D::Elem2D(int iNo, Node2D *ipN1, Node2D *ipN2, Node2D *ipN3, Node2D *ipN4, Layer2D *ipL)
{
	mNo = iNo;
	mpN1 = ipN1;
	mpN2 = ipN2;
	mpN3 = ipN3;
	mpN4 = ipN4;
	mpL = ipL;
    scaleT = 300.;
    scaleN = 1e18;
    scaleEpsR = 10.;
    scaleX = sqrt((plask::phys::epsilon0*plask::phys::kB_J*scaleT*scaleEpsR)/(plask::phys::qe*plask::phys::qe*scaleN))*1e3;
    scaleE = plask::phys::kB_eV*scaleT;
    mT = 300./scaleT;
}
//------------------------------------------------------------------------------
Elem2D::Elem2D() { }
//------------------------------------------------------------------------------
Elem2D::~Elem2D() { }
//------------------------------------------------------------------------------
void Elem2D::getInfo()
{
    std::cout << mNo << "\t" << "n1(" << mpN1->getNo() << ", " << (mpN1->getX())*scaleX << ", " << (mpN1->getY())*scaleX << ")\t"
        << "n2(" << mpN2->getNo() << ", " << (mpN2->getX())*scaleX << ", " << (mpN2->getY())*scaleX << ")\t"
        << "n3(" << mpN3->getNo() << ", " << (mpN3->getX())*scaleX << ", " << (mpN3->getY())*scaleX << ")\t"
        << "n4(" << mpN4->getNo() << ", " << (mpN4->getX())*scaleX << ", " << (mpN4->getY())*scaleX << ")\t"
        << mpL->getMaterial() << "\t" << mpL->getID() << "\t" << mpL->getMaterial() << "\n";
}
//------------------------------------------------------------------------------
int Elem2D::getNo() { return mNo; }
//------------------------------------------------------------------------------
double Elem2D::getX() { return (0.5*(mpN1->getX()+mpN2->getX())); }
//------------------------------------------------------------------------------
double Elem2D::getY() { return (0.5*(mpN1->getY()+mpN4->getY())); }
//------------------------------------------------------------------------------
double Elem2D::getSizeX() { return (mpN2->getX()-mpN1->getX()); }
//------------------------------------------------------------------------------
double Elem2D::getSizeY() { return (mpN4->getY()-mpN1->getY()); }
//------------------------------------------------------------------------------
Node2D* Elem2D::getN1Ptr() { return mpN1; }
//------------------------------------------------------------------------------
Node2D* Elem2D::getN2Ptr() { return mpN2; }
//------------------------------------------------------------------------------
Node2D* Elem2D::getN3Ptr() { return mpN3; }
//------------------------------------------------------------------------------
Node2D* Elem2D::getN4Ptr() { return mpN4; }
//------------------------------------------------------------------------------
Layer2D* Elem2D::getL() { return mpL; }
//------------------------------------------------------------------------------
double Elem2D::getT() { return mT; }
//------------------------------------------------------------------------------
double Elem2D::getFx() { return -1e-5*((0.5*(getN2Ptr()->getPsi()+getN3Ptr()->getPsi())-0.5*(getN1Ptr()->getPsi()+getN4Ptr()->getPsi()))*scaleE/(getSizeX()*scaleX)); }
//------------------------------------------------------------------------------
double Elem2D::getFy() { return -1e-5*((0.5*(getN3Ptr()->getPsi()+getN4Ptr()->getPsi())-0.5*(getN1Ptr()->getPsi()+getN2Ptr()->getPsi()))*scaleE/(getSizeY()*scaleX)); }
//------------------------------------------------------------------------------
double Elem2D::getPsi() { return 0.25*(mpN1->getPsi()+mpN2->getPsi()+mpN3->getPsi()+mpN4->getPsi()); }
//------------------------------------------------------------------------------
double Elem2D::getFn() { return 0.25*(mpN1->getFn()+mpN2->getFn()+mpN3->getFn()+mpN4->getFn()); }
//------------------------------------------------------------------------------
double Elem2D::getFp() { return 0.25*(mpN1->getFp()+mpN2->getFp()+mpN3->getFp()+mpN4->getFp()); }
//------------------------------------------------------------------------------
double Elem2D::getFnEta() { return 0.25*(mpN1->getFnEta()+mpN2->getFnEta()+mpN3->getFnEta()+mpN4->getFnEta()); }
//------------------------------------------------------------------------------
double Elem2D::getFpKsi() { return 0.25*(mpN1->getFpKsi()+mpN2->getFpKsi()+mpN3->getFpKsi()+mpN4->getFpKsi()); }
//------------------------------------------------------------------------------
double Elem2D::getN()
{
	return mN;//( mpReg->getNc(mT) * Func::IntFD12( (mFn - mPsi + mpReg->getChi()) / cPhys::kB/mT ) );
}
//------------------------------------------------------------------------------
double Elem2D::getN1()
{
	return mN1;
	//return ( func::calcN(getL()->getNcNorm(), mpN1->getFnEta(), mpN1->getPsi(), getL()->getEc0Norm(), mpN1->getT()) );
}
//------------------------------------------------------------------------------
double Elem2D::getN2()
{
	return mN2;
	//return ( func::calcN(getL()->getNcNorm(), mpN2->getFnEta(), mpN2->getPsi(), getL()->getEc0Norm(), mpN2->getT()) );
}
//------------------------------------------------------------------------------
double Elem2D::getN3()
{
	return mN3;
}
//------------------------------------------------------------------------------
double Elem2D::getN4()
{
	return mN4;
}
//------------------------------------------------------------------------------
double Elem2D::getP()
{
	return mP;//( mpReg->getNv(mT) * Func::IntFD12( ( mPsi - mpReg->getChi() - mpReg->getEg(mT) - mFp ) / cPhys::kB/mT ) ) ;
}
//------------------------------------------------------------------------------
double Elem2D::getP1()
{
	return mP1;
	//return ( func::calcP(getL()->getNvNorm(), mpN1->getFpKsi(), mpN1->getPsi(), getL()->getEv0Norm(), mpN1->getT()) );
}
//------------------------------------------------------------------------------
double Elem2D::getP2()
{
	return mP2;
	//return ( func::calcP(getL()->getNvNorm(), mpN2->getFpKsi(), mpN2->getPsi(), getL()->getEv0Norm(), mpN2->getT()) );
}
//------------------------------------------------------------------------------
double Elem2D::getP3()
{
	return mP3;
}
//------------------------------------------------------------------------------
double Elem2D::getP4()
{
	return mP4;
}
//------------------------------------------------------------------------------
double Elem2D::getJnx()
{
    return ( - getN() * getL()->getMiNNorm() * (0.5*(getN2Ptr()->getFn()+getN3Ptr()->getFn())-0.5*(getN1Ptr()->getFn()+getN4Ptr()->getFn())) / getSizeX() );
}
//------------------------------------------------------------------------------
double Elem2D::getJny()
{
    return ( - getN() * getL()->getMiNNorm() * (0.5*(getN3Ptr()->getFn()+getN4Ptr()->getFn())-0.5*(getN1Ptr()->getFn()+getN2Ptr()->getFn())) / getSizeY() );
}
//------------------------------------------------------------------------------
double Elem2D::getJpx()
{
    return ( - getP() * getL()->getMiPNorm() * (0.5*(getN2Ptr()->getFp()+getN3Ptr()->getFp())-0.5*(getN1Ptr()->getFp()+getN4Ptr()->getFp())) / getSizeX() );
}
//------------------------------------------------------------------------------
double Elem2D::getJpy()
{
    return ( - getP() * getL()->getMiPNorm() * (0.5*(getN3Ptr()->getFp()+getN4Ptr()->getFp())-0.5*(getN1Ptr()->getFp()+getN2Ptr()->getFp())) / getSizeY() );
}
//------------------------------------------------------------------------------
/*double Elem2D::getJn2()
{
    return ( - getN() * getL()->getMiNNorm() * (getN2Ptr()->getPsi()-getN1Ptr()->getPsi()) / getSizeX() + getL()->getMiNNorm() * (getN2()-getN1()) / getSizeX() );
}
//------------------------------------------------------------------------------
double Elem2D::getJp2()
{
    return ( - getP() * getL()->getMiPNorm() * (getN2Ptr()->getPsi()-getN1Ptr()->getPsi()) / getSizeX() - getL()->getMiPNorm() * (getP2()-getP1()) / getSizeX() );
}*/
//------------------------------------------------------------------------------
double Elem2D::getRsrh()
{
    return ( (getN() * getP() - getL()->getNiNorm() * getL()->getNiNorm()) / (getL()->getTpNorm() * (getN() + getL()->getNiNorm()) + getL()->getTnNorm() * (getP() + getL()->getNiNorm())) );
}
//------------------------------------------------------------------------------
double Elem2D::getRrad()
{
    return ( getL()->getBNorm() * (getN() * getP() - getL()->getNiNorm() * getL()->getNiNorm()) );
}
//------------------------------------------------------------------------------
double Elem2D::getRaug()
{
    return ( (getL()->getCnNorm() * getN() + getL()->getCpNorm() * getP()) * (getN() * getP() - getL()->getNiNorm() * getL()->getNiNorm()) );
}
//------------------------------------------------------------------------------
void Elem2D::setN(double iN)
{
	mN = iN;

	//mN = 0.5 * (getN1()+getN2());

	//mN = func::calcN(getL()->getNcNorm(), getFnEta(), getPsi(), getL()->getEc0Norm(), getT());

	/*if (mpN2->getFn() - mpN1->getFn() + mpN2->getPsi() - mpN1->getPsi())
        mN = getN1() * ((mpN2->getFn() - mpN1->getFn()) + (mpN2->getPsi() - mpN1->getPsi()))
            / (exp((mpN2->getFn() - mpN1->getFn()) + (mpN2->getPsi() - mpN1->getPsi())) - 1.);*/

    /*if (mpN1->getPsi() == mpN2->getPsi())
        mN = 0.5 * (getN1()+getN2());
    else {
        double tG = (1.-exp((mpN2->getPsi() - mpN1->getPsi())*0.5))/(1.-exp(mpN2->getPsi() - mpN1->getPsi()));
        mN = getN1()*(1.-tG) + getN2()*(tG);
    }*/

    //mN = getL()->getNcNorm() * getFnEta() * exp(getPsi()-getL()->getEc0Norm());
}
//------------------------------------------------------------------------------
void Elem2D::setP(double iP)
{
	mP = iP;

	//mP = 0.5 * (getP1()+getP2());

	// mP = func::calcP(getL()->getNvNorm(), getFpKsi(), getPsi(), getL()->getEv0Norm(), getT());

	/*if (mpN1->getFp() - mpN2->getFp() + mpN1->getPsi() - mpN2->getPsi())
        mP = getP1() * ((mpN1->getFp() - mpN2->getFp()) + (mpN1->getPsi() - mpN2->getPsi()))
            / (exp((mpN1->getFp() - mpN2->getFp()) + (mpN1->getPsi() - mpN2->getPsi())) - 1.);*/

    /*if (mpN1->getPsi() == mpN2->getPsi())
        mP = 0.5 * (getP1()+getP2());
    else {
        double tG = (1.-exp((mpN2->getPsi() - mpN1->getPsi())*0.5))/(1.-exp(mpN2->getPsi() - mpN1->getPsi()));
        mP = getP1()*(1.-tG) + getP2()*(tG);
    }*/

    //mP = getL()->getNvNorm() * getFpKsi() * exp(-getPsi()+getL()->getEv0Norm());
}
//------------------------------------------------------------------------------
void Elem2D::setN1(double iN1) { mN1 = iN1; }
//------------------------------------------------------------------------------
void Elem2D::setN2(double iN2) { mN2 = iN2; }
//------------------------------------------------------------------------------
void Elem2D::setN3(double iN3) { mN3 = iN3; }
//------------------------------------------------------------------------------
void Elem2D::setN4(double iN4) { mN4 = iN4; }
//------------------------------------------------------------------------------
void Elem2D::setP1(double iP1) { mP1 = iP1; }
//------------------------------------------------------------------------------
void Elem2D::setP2(double iP2) { mP2 = iP2; }
//------------------------------------------------------------------------------
void Elem2D::setP3(double iP3) { mP3 = iP3; }
//------------------------------------------------------------------------------
void Elem2D::setP4(double iP4) { mP4 = iP4; }
//------------------------------------------------------------------------------
void Elem2D::setT()
{
	mT = 0.25 * (mpN1->getT() + mpN2->getT() + mpN3->getT() + mpN4->getT());
}
//------------------------------------------------------------------------------
/*void Elem2D::setPsi()
{
	mPsi = 0.5 * (mpN1->getPsi() + mpN2->getPsi());
}
//------------------------------------------------------------------------------
void Elem2D::setFn()
{
	mFn = 0.5 * (mpN1->getFn() + mpN2->getFn());
}
//------------------------------------------------------------------------------
void Elem2D::setFp()
{
	mFp = 0.5 * (mpN1->getFp() + mpN2->getFp());
}*/
