#include "node2D.h"

namespace plask { namespace modules { namespace thermal {

//CONSTRUCTORS-----------------------------------------------------------------------------------------------------------
//Constructor - set Numer, Coordinates Pointers, Layer Pointer or NULL if Boundary, Temperature (Const)
Node2D::Node2D(int iNr, const double* ipX, const double* ipY, const Layer* ipLayer, bool ifTConst, double iT):
mNr(iNr), mpX(ipX), mpY(ipY), mpLayer(ipLayer), mfTConst(ifTConst), mT(iT), mPsi(0.0), mFn(0.0), mFp(0.0)
{
}


//GETTERS----------------------------------------------------------------------------------------------------------------
int Node2D::getNr() const  { return mNr; } //Return Numer
const Layer* Node2D::getLayerPtr() const  { return mpLayer; } //Return Layer Pointer or NULL if Boundary
double Node2D::getX() const  { return *mpX; } //Return X Coordinate
double Node2D::getY() const  { return *mpY; } //Return Y Coordinate
const double* Node2D::getXPtr() const  { return mpX; } //Return X Coordinate Pointer
const double* Node2D::getYPtr() const  { return mpY; } //Return Y Coordinate Pointer
double Node2D::getT() const   { return mT; } //Return Temperature
bool Node2D::ifTConst() const  { return mfTConst; } //Check if Constant Temperature

//Return Settings
std::string Node2D::getSetUp()
{
	return ( (boost::format("%.0f") % mNr).str() + " \t" + (mpLayer ? mpLayer->getName() : "boundary") + " \t" +
			  mVolContSide + " \t" + (mfTConst ? "true" : "false") + " \t" + (boost::format("%.4f") % (*mpX)).str() + " \t" +
			 (boost::format("%.4f") % (*mpY)).str() + " \t" );
}

//Return Data
std::string Node2D::getData()
{
	return ( (boost::format("%.0f") % mNr).str() + " \t" + (boost::format("%.6e") % mT).str() + " \t" +
			 (boost::format("%.6e") % mPsi).str() + " \t" + (boost::format("%.6e") % mFn).str() + " \t" +
			 (boost::format("%.6e") % mFp).str() + " \t" );
}


//SETTERS----------------------------------------------------------------------------------------------------------------
void Node2D::setT(double iT)  { if (!mfTConst)  mT = iT; } //Set Temperature

}}} // namespace plask::modules::thermal