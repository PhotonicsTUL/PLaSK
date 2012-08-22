#include "node2D.h"

namespace plask { namespace solvers { namespace thermal {

Node2D::Node2D(int iNo, const double* ipX, const double* ipY, double iT,  bool ifTConst):
mNo(iNo), mpX(ipX), mpY(ipY), mT(iT), mfTConst(ifTConst)
{
}

int Node2D::getNo() const { return mNo; }
double Node2D::getX() const { return *mpX; }
double Node2D::getY() const { return *mpY; }
double Node2D::getT() const { return mT; }
bool Node2D::ifTConst() const { return mfTConst; }
const double* Node2D::getXPtr() const { return mpX; }
const double* Node2D::getYPtr() const { return mpY; }

void Node2D::setT(double iT)  { if (!mfTConst)  mT = iT; }

}}} // namespace plask::solvers::thermal
