#ifndef PLASK__AlxOy_H
#define PLASK__AlxOy_H

/** @file
This file includes AlxOy
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent AlxOy, its physical properties.
 */
struct AlxOy: public Dielectric {

    static constexpr const char* NAME = "AlxOy";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};


} // namespace plask

#endif	//PLASK__AlxOy_H
