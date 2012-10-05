#ifndef PLASK__InP_H
#define PLASK__InP_H

/** @file
This file includes undoped InP
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped InP, its physical properties.
 */
struct InP: public Semiconductor {

    static constexpr const char* NAME = "InP";

    virtual std::string name() const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;

};

} // namespace plask

#endif	//PLASK__InP_H
