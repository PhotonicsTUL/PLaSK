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
bool Node2D::ifTConst() const { return mfTConst; }
bool Node2D::ifHFConst() const { return mfHFConst; }

void Node2D::setT(double iT)  { if (!mfTConst)  mT = iT; }
void Node2D::setHF(double iHF)  { mHF = iHF; }
void Node2D::setHFflag(bool ifHF) { mfHFConst = ifHF; }

}}} // namespace plask::solvers::thermal
