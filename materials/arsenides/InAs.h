#ifndef PLASK__InAs_H
#define PLASK__InAs_H

/** @file
This file includes undoped InAs
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped InAs, its physical properties.
 */
struct InAs: public Semiconductor {

    static constexpr const char* NAME = "InAs";

    virtual std::string name() const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;

};


} // namespace plask

#endif	//PLASK__InAs_H
