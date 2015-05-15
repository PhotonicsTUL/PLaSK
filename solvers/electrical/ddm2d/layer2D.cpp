#include "layer2D.h"
//------------------------------------------------------------------------------
Layer2D::Layer2D(int iNo, double iThicknessX, double iThicknessY, std::string iMaterial, double iX, double iY, double iZ, std::string iDopant, double iDopConc, std::string iID)
{
	mNo = iNo;
	mThicknessX = iThicknessX;
	mThicknessY = iThicknessY;
	mMaterial = iMaterial;
	mX = iX;
	mY = iY;
	mZ = iZ;
	mDopant = iDopant;
	mDopConc = iDopConc;
	mID = iID;
    scaleT = 300.;
    scaleE = plask::phys::kB_eV*scaleT;
    scaleN = 1e18;
    scaleEpsR = 10.;
    scaleX = sqrt((plask::phys::epsilon0*plask::phys::kB_J*scaleT*scaleEpsR)/(plask::phys::qe*plask::phys::qe*scaleN))*1e3;
    scaleK = 100.;
    scaleMi = 1000.;
    scaleR = ((plask::phys::kB_J*scaleT*scaleMi*scaleN)/(plask::phys::qe*scaleX*scaleX))*1e8;
    scalet = scaleN/scaleR;
    scaleB = scaleR/(scaleN*scaleN);
    scaleC = scaleR/(scaleN*scaleN*scaleN);
    scaleH = ((scaleK*scaleT)/(scaleX*scaleX))*1e12;
}
//------------------------------------------------------------------------------
Layer2D::Layer2D() { }
//------------------------------------------------------------------------------
Layer2D::~Layer2D()
{
    //delete mpDB;
    //mpDB = NULL;
}
//------------------------------------------------------------------------------
void Layer2D::getInfo()
{
	std::cout << mNo << "\t" << getThicknessX() << "\t" << getThicknessY() << "\t" << getMaterial() << "\t" << mX << "\t" << mY << "\t" << mZ << "\t" << mDopant << "\t" << mDopConc << "\t" << mID << "\n";
}
//------------------------------------------------------------------------------
int Layer2D::getNo() { return mNo; }
//------------------------------------------------------------------------------
double Layer2D::getThicknessX() { return mThicknessX; }
//------------------------------------------------------------------------------
double Layer2D::getThicknessY() { return mThicknessY; }
//------------------------------------------------------------------------------
std::string Layer2D::getMaterial() { return mMaterial; }
//------------------------------------------------------------------------------
double Layer2D::getX() { return mX; }
//------------------------------------------------------------------------------
double Layer2D::getY() { return mY; }
//------------------------------------------------------------------------------
double Layer2D::getZ() { return mZ; }
//------------------------------------------------------------------------------
std::string Layer2D::getDopant() { return mDopant; }
//------------------------------------------------------------------------------
double Layer2D::getDopConc() { return mDopConc; }
//------------------------------------------------------------------------------
std::string Layer2D::getID() { return mID; }
//------------------------------------------------------------------------------
double Layer2D::getNd()
{
    if (((getMaterial() == "GaAs")||(getMaterial() == "AlGaAs"))&&(getDopant() == "C")) return 0.; // Nd (cm^-3)
    else if (((getMaterial() == "GaAs")||(getMaterial() == "AlGaAs"))&&(getDopant() == "Si")) return mDopConc; // Nd (cm^-3)
    else if (((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN"))&&(getDopant() == "Mg")) return 0.; // Nd (cm^-3)
    else if (((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN"))&&(getDopant() == "Si")) return mDopConc; // Nd (cm^-3)
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getNdNorm() // normalized donor concentration (-)
{
    return ( (getNd())/scaleN );
}
//------------------------------------------------------------------------------
double Layer2D::getNa()
{
    if (((getMaterial() == "GaAs")||(getMaterial() == "AlGaAs"))&&(getDopant() == "C")) return mDopConc; // Na (cm^-3)
    else if (((getMaterial() == "GaAs")||(getMaterial() == "AlGaAs"))&&(getDopant() == "Si")) return 0.; // Na (cm^-3)
    else if (((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN"))&&(getDopant() == "Mg")) return mDopConc; // Na (cm^-3)
    else if (((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN"))&&(getDopant() == "Si")) return 0.; // Na (cm^-3)
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getNaNorm() // normalized acceptor concentration (-)
{
    return ( (getNa())/scaleN );
}
//------------------------------------------------------------------------------
double Layer2D::getK() // thermal conductivity (-)
{
    if (mMaterial == "GaAs") return (100./2.27); //TODO
    else if (mMaterial == "AlGaAs") return (100./(2.27+28.83*mX-30.0*mX*mX)); //TODO
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (150.); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getKNorm() // normalized thermal conductivity (-)
{
    return ( (getK())/scaleK );
}
//------------------------------------------------------------------------------
double Layer2D::getEpsR() // dielectric constant (-)
{
    if (mMaterial == "GaAs") return 12.90;
    else if (mMaterial == "AlGaAs") return (12.90-2.84*mX);
    else {
        double tAlN = 8.5, tGaN = 8.9, tInN = 15.3;
        if (mMaterial == "GaN") return tGaN;
        else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
        else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
        else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
        else return -1;
    }
}
//------------------------------------------------------------------------------
double Layer2D::getEpsRNorm() // normalized dielectric constant (-)
{
    return ( (getEpsR())/scaleEpsR );
}
//------------------------------------------------------------------------------
double Layer2D::getEg() // energy gap (eV)
{
    if (mMaterial == "GaAs") return 1.424;
    else if (mMaterial == "AlGaAs")
    {
        if (mX < 0.45) return (1.424+1.247*mX);
        else return (1.9+0.125*mX+0.143*mX*mX);
    }
    else {
        double tAlN = 6.2, tGaN = 3.4, tInN = 0.7;
        if (mMaterial == "GaN") return tGaN;
        else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN-mX*(1.-mX)*1.0);
        else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN-mX*(1.-mX)*1.2);
        else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN-mX*mY*1.0-mX*(1.-mX-mY)*4.5-mY*(1.-mX-mY)*1.2);
        else return -1;
    }
}
//------------------------------------------------------------------------------
double Layer2D::getEgNorm() // normalized energy gap (-)
{
    return ( (getEg())/scaleE );
}
//------------------------------------------------------------------------------
double Layer2D::getEc0() // conduction band edge (eV)
{
    return ( getEv0()+getEg() );
}
//------------------------------------------------------------------------------
double Layer2D::getEc0Norm() // normalized conduction band edge (-
{
    return ( (getEc0())/scaleE );
}
//------------------------------------------------------------------------------
double Layer2D::getEv0() // valence band edge (eV)
{
    if (mMaterial == "GaAs") return -0.080;
    else if (mMaterial == "AlGaAs") return ( mX*(-1.33)+(1.-mX)*(-0.080) );
    else {
        double tAlN = -0.2, tGaN = 0., tInN = 0.5;
        if (mMaterial == "GaN") return tGaN;
        else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
        else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
        else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
        else return -1;
    }
}
//------------------------------------------------------------------------------
double Layer2D::getEv0Norm() // normalized valence band edge (-)
{
    return ( (getEv0())/scaleE );
}
//------------------------------------------------------------------------------
double Layer2D::getMe() // electron effective mass (m0)
{
    if (mMaterial == "GaAs") return 0.063;
    else if (mMaterial == "AlGaAs") return (0.063+0.083*mX);
    else {
        double tAlN = 0.25, tGaN = 0.2, tInN = 0.1;
        if (mMaterial == "GaN") return tGaN;
        else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
        else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
        else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
        else return -1;
    }
}
//------------------------------------------------------------------------------
double Layer2D::getMh() // hole effective mass (m0)
{
    if (mMaterial == "GaAs") return 0.51;
    else if (mMaterial == "AlGaAs") return (0.51+0.25*mX);
    else {
        double tAlN = 0.5, tGaN = 0.5, tInN = 0.5; // TODO
        if (mMaterial == "GaN") return tGaN;
        else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
        else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
        else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
        else return -1;
    }
}
//------------------------------------------------------------------------------
double Layer2D::getNi() // electron density of states (cm^-3)
{
    return ( sqrt(getNc()*getNv())*exp(-0.5*getEg()/(plask::phys::kB_eV*300.)) );
}
//------------------------------------------------------------------------------
double Layer2D::getNiNorm() // normalized electron density of states (-)
{
    return ( (getNi())/scaleN );
}
//------------------------------------------------------------------------------
double Layer2D::getNc() // electron density of states (cm^-3)
{
    return ( 2e-6*pow((getMe()*plask::phys::me*plask::phys::kB_eV*300.)/(2.*M_PI*plask::phys::hb_eV*plask::phys::hb_J),1.5) );
}
//------------------------------------------------------------------------------
double Layer2D::getNcNorm() // normalized electron density of states (-)
{
    return ( (getNc())/scaleN );
}
//------------------------------------------------------------------------------
double Layer2D::getNv() // hole density of states (cm^-3)
{
    return ( 2e-6*pow((getMh()*plask::phys::me*plask::phys::kB_eV*300.)/(2.*M_PI*plask::phys::hb_eV*plask::phys::hb_J),1.5) );
}
//------------------------------------------------------------------------------
double Layer2D::getNvNorm() // normalized hole density of states (-)
{
    return ( (getNv())/scaleN );
}
//------------------------------------------------------------------------------
double Layer2D::getEd() // donor ionization energy (meV) // TODO
{
    if (mMaterial == "GaAs") return 3.;
    else if (mMaterial == "AlGaAs") return 5.;
    else if (mMaterial == "GaN") return 13.;
    else if (mMaterial == "AlGaN") return 13.;
    else if (mMaterial == "InGaN") return 13.;
    else if (mMaterial == "AlInGaN") return 13.;
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getEdNorm() // normalized donor ionization energy (-)
{
    return ( (1e-3*getEd())/scaleE );
}
//------------------------------------------------------------------------------
double Layer2D::getEa() // acceptor ionization energy (meV) // TODO
{
    if (mMaterial == "GaAs") return 10.;
    else if (mMaterial == "AlGaAs") return 50.;
    else if (mMaterial == "GaN") return 170.;
    else if (mMaterial == "AlGaN") return 170.;
    else if (mMaterial == "InGaN") return 170.;
    else if (mMaterial == "AlInGaN") return 170.;
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getEaNorm() // normalized acceptor ionization energy (-)
{
    return ( (1e-3*getEa())/scaleE );
}
//------------------------------------------------------------------------------
double Layer2D::getMiN() // electron mobility (cm^2/Vs)
{
    if (mMaterial == "GaAs") return 8000.;
    else if (mMaterial == "AlGaAs")
    {
        if (mX < 0.45) return (8000.+22000.*mX+10000.*mX*mX);
        else return (255.+1160.*mX-720.*mX*mX);
    }
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (2000.); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getMiNNorm() // normalized electron mobility (-)
{
    return ( (getMiN())/scaleMi );
}
//------------------------------------------------------------------------------
double Layer2D::getMiP() // hole mobility (cm^2/Vs)
{
    if (mMaterial == "GaAs") return 370.;
    else if (mMaterial == "AlGaAs") return (370.-970.*mX+740.*mX*mX);
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (200.); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getMiPNorm() // normalized hole mobility (-)
{
    return ( (getMiP())/scaleMi );
}
//------------------------------------------------------------------------------
double Layer2D::getTn() // SRH recombination lifetime for electrons (s)
{
    if (mMaterial == "GaAs") return 1e-7;
    else if (mMaterial == "AlGaAs") return 1e-8;
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (1e-7); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getTnNorm() // normalized SRH recombination lifetime for electrons (-)
{
    return ( (getTn())/scalet );
}
//------------------------------------------------------------------------------
double Layer2D::getTp() // SRH recombination lifetime for holes (s)
{
    if (mMaterial == "GaAs") return 1e-7;
    else if (mMaterial == "AlGaAs") return 1e-8;
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (1e-7); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getTpNorm() // normalized SRH recombination lifetime for holes (-)
{
    return ( (getTp())/scalet );
}
//------------------------------------------------------------------------------
double Layer2D::getB() // radiative recombination coefficient (cm^3/s)
{
    if (mMaterial == "GaAs") return 2.04e-10;
    else if (mMaterial == "AlGaAs") return 2.04e-10;
    else if ((getMaterial() == "GaN")||(getMaterial() == "AlGaN")||(getMaterial() == "InGaN")||(getMaterial() == "AlInGaN")) return (2.4e-11); //TODO
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getBNorm() // normalized radiative recombination coefficient (-)
{
    return ( (getB())/scaleB );
}
//------------------------------------------------------------------------------
double Layer2D::getCn() // Auger recombination coefficient for electrons (cm^6/s)
{
    if (mMaterial == "GaAs") return 1.6e-29;
    else if (mMaterial == "AlGaAs") return 1.6e-29;
    else if ((mMaterial == "GaN")||(mMaterial == "AlGaN")||(mMaterial == "InGaN")||(mMaterial == "AlInGaN")) return 0.7e-31;
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getCnNorm() // normalized Auger recombination coefficient for electrons
{
    return ( (getCn())/scaleC );
}
//------------------------------------------------------------------------------
double Layer2D::getCp() // Auger recombination coefficient for holes (cm^6/s)
{
    if (mMaterial == "GaAs") return 4.64e-29;
    else if (mMaterial == "AlGaAs") return 4.64e-29;
    else if ((mMaterial == "GaN")||(mMaterial == "AlGaN")||(mMaterial == "InGaN")||(mMaterial == "AlInGaN")) return 1.4e-31;
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getCpNorm() // normalized Auger recombination coefficient for holes
{
    return ( (getCp())/scaleC );
}
//------------------------------------------------------------------------------
double Layer2D::getH(double iT) // heat source (W/m^3)
{
    if (mMaterial == "GaAs") return 0.;
    else if (mMaterial == "AlGaAs") return 0.;//(1e3/300)*iT; // 1e3
    else if ((mMaterial == "GaN")||(mMaterial == "AlGaN")||(mMaterial == "InGaN")||(mMaterial == "AlInGaN")) return 0.;
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getHNorm(double iT) // normalized heat source
{
    return ( (getH(iT))/scaleH );
}
//------------------------------------------------------------------------------
double Layer2D::getPsp() // get spontaneous polarization (C/m^2)
{
    double tAlN = -0.081, tGaN = -0.029, tInN = -0.032;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getAlc() // get lattice constant (A)
{
    double tAlN = 3.112, tGaN = 3.188, tInN = 3.540;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::gete13() // get piezoelectric constant (C/m^2)
{
    double tAlN = -0.58, tGaN = -0.33, tInN = -0.22;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::gete33() // get piezoelectric constant (C/m^2)
{
    double tAlN = 1.55, tGaN = 0.65, tInN = 0.43;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getC13() // get elastic constant (GPa)
{
    double tAlN = 115., tGaN = 105., tInN = 95.;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
double Layer2D::getC33() // get elastic constant (GPa)
{
    double tAlN = 385., tGaN = 395., tInN = 200.;
    if (mMaterial == "GaN") return tGaN;
    else if (mMaterial == "AlGaN") return (mX*tAlN+(1.-mX)*tGaN);
    else if (mMaterial == "InGaN") return (mX*tInN+(1.-mX)*tGaN);
    else if (mMaterial == "AlInGaN") return (mX*tAlN+mY*tGaN+(1.-mX-mY)*tInN);
    else return -1;
}
//------------------------------------------------------------------------------
