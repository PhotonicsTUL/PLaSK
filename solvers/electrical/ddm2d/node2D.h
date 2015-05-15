/**************************************************************************************************
									PREPROCESSOR DIRECTIVES
**************************************************************************************************/
#ifndef NODE2D_H
#define NODE2D_H

/**************************************************************************************************
										INCLUDES
**************************************************************************************************/
//#include "headers.h"
//#include "DDMesh1D.h"
#include <iostream>
#include "../../../plask/phys/constants.h"
#include <cmath>

/**************************************************************************************************
										CLASS Node2D
**************************************************************************************************/
class Node2D
{
public:
	Node2D(int iNo, double iX, double iY, char iSide);
	Node2D();
	~Node2D();

	// getters
	void getInfo(); /// all info about the node
	int getNo();
	char getSide();
	double getX();
	double getY();
	double getT();
	double getPsi();
	double getFn();
	double getFp();
	double getFnEta();
	double getFpKsi();
    int getFlagPsi0();
	void setT(double iT);
	void setPsi(double iPsi);
	void setFn(double iFn);
	void setFp(double iFp);
	void setFnEta(double iFnEta);
	void setFpKsi(double iFpKsi);
	void setFlagPsi0(int ifPsi0);

private:
	// members
	int mNo; // node number
	char mSide;
    double mX; // x-coordinate
    double mY; // y-coordinate
	double mT;
	double mPsi;
	double mFn;
	double mFp;
	double mFnEta;
	double mFpKsi;
	int mfPsi0; // >0 if initial potential has been set (value = how many times)
    double scaleX, scaleT, scaleN, scaleEpsR; // scaling parameters for position (in [um]), temperature (in [K])
};

/**************************************************************************************************
									PREPROCESSOR DIRECTIVE
**************************************************************************************************/
#endif // Node2D_H
