#ifndef PLASK__MODULE_THERMAL_NODE2D_H
#define PLASK__MODULE_THERMAL_NODE2D_H

// #include "../Headers.h"
// #include "Layer.h"
namespace plask { namespace modules { namespace thermal {


class Node2D //Node of Finite Element Model
{
public:
    //Constructor - set Numer, Coordinates Pointers, Temperature (Const)
    Node2D(int iNr, const double* ipX, const double* ipY, bool ifTConst = false);

	//Getters
    int getNr() const; //Return Numer
    double getX() const; //Return X Coordinate
    double getY() const; //Return Y Coordinate
    const double* getXPtr() const; //Return X Coordinate Pointer
    const double* getYPtr() const; //Return Y Coordinate Pointer
    double getT() const; //Return Temperature
	bool ifTConst() const; //Check if Constant Temperature
	std::string getSetUp(); //Return Set-Up
	std::string getData(); //Return Data

	//Setters
	void setT(double iT); //Set Temperature

protected:
    //Members
	int mNr; //Numer
	const Layer* mpLayer; //Layer Pointer or NULL if Boundary
	bool mfTConst; //If Const Temperature
    const double *mpX, *mpY; //Pointers to X, Y Coordinates
    double mT; //Temperature
};

}}} // namespace plask::modules::thermal
#endif
