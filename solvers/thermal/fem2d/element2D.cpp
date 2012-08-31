#include "element2D.h"

namespace plask { namespace solvers { namespace thermal {

Element2D::Element2D(int iNo, const Node2D* ipNLoLeft, const Node2D* ipNLoRight, const Node2D* ipNUpLeft, const Node2D* ipNUpRight):
mNo(iNo), mpNLoLeft(ipNLoLeft), mpNLoRight(ipNLoRight), mpNUpLeft(ipNUpLeft), mpNUpRight(ipNUpRight)
{
}

int Element2D::getNo() const { return mNo; }
const Node2D* Element2D::getNLoLeftPtr() const { return mpNLoLeft; }
const Node2D* Element2D::getNLoRightPtr() const { return mpNLoRight; }
const Node2D* Element2D::getNUpLeftPtr()  const { return mpNUpLeft; }
const Node2D* Element2D::getNUpRightPtr() const { return mpNUpRight; }
double Element2D::getWidth() const { return (mpNUpRight->getX() - mpNUpLeft->getX()); }
double Element2D::getHeight() const { return (mpNUpLeft->getY() - mpNLoLeft->getY()); }
double Element2D::getT() const { return mT; }

void Element2D::setT()
{
	mT = (mpNLoLeft->getT() + mpNLoRight->getT() + mpNUpLeft->getT() + mpNUpRight->getT()) * 0.25;
}

}}} // namespace plask::solvers::thermal
