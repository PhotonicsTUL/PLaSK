#ifndef PLASK__AlAs_H
#define PLASK__AlAs_H

/** @file
This file includes undoped AlAs
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped AlAs, its physical properties.
 */
struct AlAs: public Semiconductor {

    static constexpr const char* NAME = "AlAs";

    virtual std::string name() const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;

};


} // namespace plask

#endif	//PLASK__AlAs_H
