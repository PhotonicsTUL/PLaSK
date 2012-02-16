#ifndef PLASK__InN_H
#define PLASK__InN_H

/** @file
This file includes undoped InN
*/

#include "../material.h"


namespace plask {

/**
 * Represent undoped InN, its physical properties.
 */
struct InN: public Material {

    static constexpr const char* NAME = "InN";

    InN();
    virtual std::string name() const;
    virtual double condT(double T) const; //NO Temperature dependence
private:
    double condTmax_RT;
};


} // namespace plask

#endif	//PLASK__InN_H
