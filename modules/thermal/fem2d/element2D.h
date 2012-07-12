#ifndef PLASK__MODULE_THERMAL_ELEMENT2D_H
#define PLASK__MODULE_THERMAL_ELEMENT2D_H

#include <string>

#include "node2D.h"

namespace plask { namespace modules { namespace thermal {

class Element2D //Element of Finite Element Model
{
public:
	//Constructor - set Numer, Layer Pointer and Four Nodes Pointers
    Element2D(int iNr, const Layer* ipL, const Node2D* ipNLoLeft, const Node2D* ipNLoRight, const Node2D* ipNUpLeft, const Node2D* ipNUpRight);

    //Getters
    int getNr() const; //Return Numer
    const Node2D* getNLoLeftPtr() const; //Return Lower Left Node Pointer
    const Node2D* getNLoRightPtr() const; //Return Lower Right Node Pointer
    const Node2D* getNUpLeftPtr() const; //Return Upper Left Node Pointer
    const Node2D* getNUpRightPtr() const; //Return Upper Right Node Pointer
    const Layer* getLPtr() const; //Return Layer Pointer
	double getWidth() const; //Return Width
	double getHeight() const; //Return Hight
	double getT() const; //Return Temperature
	std::string getSetUp(); //Return Set-Up

	//Setters
    void setTAver(); //Set Temperature From Nodes

protected:
	//Members
    int mNr; //Number
    const Layer *mpL; //Pointer to Layer
    const Node2D *mpNLoLeft, //Pointer to Lower Left Node
		       *mpNLoRight, //Pointer to Lower Right Node
			   *mpNUpLeft, //Pointer to Upper Left Node
			   *mpNUpRight; //Pointer to Upper Right Node
    double mT, //Average Temperature From Nodes
};

}}} // namespace plask::modules::thermal
#endif
