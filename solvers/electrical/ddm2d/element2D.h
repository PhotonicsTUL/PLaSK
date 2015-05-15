/**************************************************************************************************
									PREPROCESSOR DIRECTIVES
**************************************************************************************************/
#ifndef ELEMENT2D_H
#define ELEMENT2D_H

/**************************************************************************************************
										INCLUDES
**************************************************************************************************/
#include "../../../plask/phys/constants.h"
#include <iostream>
#include "node2D.h"
#include "layer2D.h"

/**************************************************************************************************
										CLASS ELEMENT1D
**************************************************************************************************/
class Elem2D
{
public:
	Elem2D(int iNo, Node2D *ipN1, Node2D *ipN2, Node2D *ipN3, Node2D *ipN4, Layer2D *ipL);
	Elem2D();
	~Elem2D();

	// getters
	void getInfo(); /// all info about the element
	int getNo();
	double getX(); /// normalised x-position of the center of the element (-)
	double getY(); /// normalised y-position of the center of the element (-)
	double getSizeX();
	double getSizeY();
	Node2D* getN1Ptr();
	Node2D* getN2Ptr();
	Node2D* getN3Ptr();
	Node2D* getN4Ptr();
	Layer2D* getL(); /// get layer pointer
	double getT();
	double getFx(); /// get electric field (MV/cm)
	double getFy(); /// get electric field (MV/cm)
	double getPsi();
	double getFn();
	double getFp();
	double getFnEta();
	double getFpKsi();
	double getN();
	double getN1(); /// electron concentration in node 1
	double getN2(); /// electron concentration in node 2
	double getN3(); /// electron concentration in node 3
	double getN4(); /// electron concentration in node 4
	double getP();
	double getP1(); /// hole concentration in node 1
	double getP2(); /// hole concentration in node 2
	double getP3(); /// hole concentration in node 3
	double getP4(); /// hole concentration in node 4
	double getJnx(); /// normalised JnX (-)
	double getJny(); /// normalised JnY (-)
	double getJpx(); /// normalised JpX (-)
	double getJpy(); /// normalised JpY (-)
	//double getJn2(); /// normalised Jn (-) calculated with the use of potential and concentration, not quasi-Fermi levels
	//double getJp2(); /// normalised Jp (-) calculated with the use of potential and concentration, not quasi-Fermi levels
	double getRsrh(); /// normalised Rsrh (-)
	double getRrad(); /// normalised Rrad (-)
	double getRaug(); /// normalised Raug (-)

    // setters
	void setT();
	/*void setPsi();
	void setFn();
	void setFp();*/
	void setN(double iN);
	void setP(double iP);
	void setN1(double iN1);
    void setN2(double iN2);
	void setN3(double iN3);
    void setN4(double iN4);
	void setP1(double iP1);
	void setP2(double iP2);
	void setP3(double iP3);
	void setP4(double iP4);

private:
    // members
	Node2D *mpN1, *mpN2, *mpN3, *mpN4;
	Layer2D *mpL;
	int mNo;
	double mT;
	//double mPsi, mFn, mFp;
	double mN, mP;
	double mN1, mN2, mN3, mN4, mP1, mP2, mP3, mP4;
    double scaleE, scaleX, scaleT, scaleN, scaleEpsR; // scaling parameters for energy (in [eV]), position (in [um]), temperature (in [K]), concentration (in [1/cm^3]), dielectric constant (no unit)
};

/**************************************************************************************************
									PREPROCESSOR DIRECTIVE
**************************************************************************************************/
#endif // Elem2D_H

