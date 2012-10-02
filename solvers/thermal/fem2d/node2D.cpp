#include "node2D.h"

namespace plask { namespace solvers { namespace thermal {

Node2D::Node2D(int iNo, double iX, double iY, double iT,  bool ifTConst):
mNo(iNo), mX(iX), mY(iY), mT(iT), mfTConst(ifTConst)
{
}

int Node2D::getNo() const { return mNo; }
double Node2D::getX() const { return mX; }
double Node2D::getY() const { return mY; }
double Node2D::getT() const { return mT; }
double Node2D::getHF() const { return mHF; }
double Node2D::getConvCoeff() const { return mConvCoeff; }
double Node2D::getTAmb1() const { return mTAmb1; }
double Node2D::getEmissivity() const { return mEmissivity; }
double Node2D::getTAmb2() const { return mTAmb2; }
bool Node2D::ifTConst() const { return mfTConst; }
bool Node2D::ifHFConst() const { return mfHFConst; }
bool Node2D::ifConvection() const { return mfConvection; }
bool Node2D::ifRadiation() const { return mfRadiation; }

void Node2D::setT(double iT)  { if (!mfTConst)  mT = iT; }
void Node2D::setHF(double iHF)  { mHF = iHF; }
void Node2D::setHFflag(bool ifHF) { mfHFConst = ifHF; }
void Node2D::setConv(double iConvCoeff, double iTAmb1) { mConvCoeff = iConvCoeff; mTAmb1 = iTAmb1; }
void Node2D::setConvflag(bool ifConvection) { mfConvection = ifConvection; }
void Node2D::setRad(double iEmissivity, double iTAmb2) { mEmissivity = iEmissivity; mTAmb2 = iTAmb2; }
void Node2D::setRadflag(bool ifRadiation) { mfRadiation = ifRadiation; }

}}} // namespace plask::solvers::thermal
