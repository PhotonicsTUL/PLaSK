#ifndef PLASK__Ni_H
#define PLASK__Ni_H

/** @file
This file includes Ni
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Ni, its physical properties.
 */
struct Ni: public Metal {

    static constexpr const char* NAME = "Ni";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};


} // namespace plask

#endif	//PLASK__Ni_H
