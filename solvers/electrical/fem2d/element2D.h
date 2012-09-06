#ifndef PLASK__MODULE_ELECTRICAL_ELEMENT2D_H
#define PLASK__MODULE_ELECTRICAL_ELEMENT2D_H

#include <string>
#include "node2D.h"

namespace plask { namespace solvers { namespace electrical {

class Element2D // 2D element in finite element method
{
public:
    Element2D(int iNo, const Node2D* ipNLoLeft, const Node2D* ipNLoRight, const Node2D* ipNUpLeft, const Node2D* ipNUpRight); // constructor (set no, nodes pointers)

    // getters
    int getNo() const; // return number
    const Node2D* getNLoLeftPtr() const; // return pointer to bottom-left node
    const Node2D* getNLoRightPtr() const; // return pointer to bottom-right node
    const Node2D* getNUpLeftPtr() const; // return pointer to top-left node
    const Node2D* getNUpRightPtr() const; // return pointer to top-right node
    double getWidth() const; // return width
    double getHeight() const; // return height
    //double getT() const; // return temperature

    // setters
    //void setT(); // set temperature (from nodes)

protected:
    int mNo; // number
    const Node2D *mpNLoLeft, // pointer to bottom-left node
               *mpNLoRight, // pointer to bottom-right node
               *mpNUpLeft, // pointer to top-left node
               *mpNUpRight; // pointer to top-right nodeRight Node
    //double mT; // average temperature
};

}}} // namespace plask::solvers::electrical
#endif
