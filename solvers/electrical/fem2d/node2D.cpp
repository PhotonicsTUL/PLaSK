#include "node2D.h"

namespace plask { namespace solvers { namespace electrical {

Node2D::Node2D(int iNo, double iX, double iY, double iV,  bool ifVConst):
mNo(iNo), mX(iX), mY(iY), mV(iV), mfVConst(ifVConst)
{
}

int Node2D::getNo() const { return mNo; }
double Node2D::getX() const { return mX; }
double Node2D::getY() const { return mY; }
double Node2D::getV() const { return mV; }
bool Node2D::ifVConst() const { return mfVConst; }

void Node2D::setV(double iV)  { if (!mfVConst)  mV = iV; }

}}} // namespace plask::solvers::electrical
