#include "element2D.h"

namespace plask { namespace solvers { namespace thermal {

//CONSTRUCTORS-----------------------------------------------------------------------------------------------------------
//Constructor - set Numer, Layer Pointer and Four Nodes Pointers
Element2D::Element2D(int iNo, /*const Layer* ipL,*/ const Node2D* ipNLoLeft, const Node2D* ipNLoRight, const Node2D* ipNUpLeft, const Node2D* ipNUpRight, double iT):
mNo(iNo), /*mpL(ipL),*/ mpNLoLeft(ipNLoLeft), mpNLoRight(ipNLoRight), mpNUpLeft(ipNUpLeft), mpNUpRight(ipNUpRight),
mT(iT)
{
}


//GETTERS----------------------------------------------------------------------------------------------------------------
int Element2D::getNo() const  { return mNo; } //Return Numer
const Node2D* Element2D::getNLoLeftPtr() const  { return mpNLoLeft; } //Return Lower Left Node Pointer
const Node2D* Element2D::getNLoRightPtr() const  { return mpNLoRight; } //Return Lower Right Node Pointer
const Node2D* Element2D::getNUpLeftPtr()  const  { return mpNUpLeft; } //Return Upper Left Node Pointer
const Node2D* Element2D::getNUpRightPtr() const  { return mpNUpRight; } //Return Upper Right Node Pointer
//const Layer* Element::getLPtr() const  { return mpL; } //Return Layer Pointer
double Element2D::getT() const  { return mT; } //Return Temperature
double Element2D::getWidth() const  { return (mpNUpRight->getX() - mpNUpLeft->getX()); } //Return Width
double Element2D::getHeight() const  { return (mpNUpLeft->getY() - mpNLoLeft->getY()); } //Return Height

//Return Set-Up
/*std::string Element::getSetUp()
{
	return ( (boost::format("%.0f") % mNr).str() + " \t" + mpL->getName() + " \t" + (boost::format("%.0f") % mpNLoLeft->getNr()).str() + " \t" +
			 (boost::format("%.0f") % mpNLoRight->getNr()).str() + " \t" + (boost::format("%.0f") % mpNUpLeft->getNr()).str() + " \t" +
			 (boost::format("%.0f") % mpNUpRight->getNr()).str() + "\t" + (mpELeft ? (boost::format("%.0f") % mpELeft->getNr()).str() : "0") + "\t" +
			 (mpERight ? (boost::format("%.0f") % mpERight->getNr()).str() : "0") + "\t" + (mpELo ? (boost::format("%.0f") % mpELo->getNr()).str() : "0") + "\t" +
			 (mpEUp ? (boost::format("%.0f") % mpEUp->getNr()).str() : "0") + "\t" );
}*/

//SETTERS----------------------------------------------------------------------------------------------------------------

//Set Temperature
void Element2D::setT()
{
	mT = (mpNLoLeft->getT() + mpNLoRight->getT() + mpNUpLeft->getT() + mpNUpRight->getT()) * 0.25;
    //if (Func::isWrongVal(mT)) throw ElementErr("setT", "Temperature", mT, mNr);
}

}}} // namespace plask::solvers::thermal
