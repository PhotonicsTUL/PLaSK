#ifndef PLASK__AuZn_H
#define PLASK__AuZn_H

/** @file
This file includes AuZn
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent AuZn, its physical properties.
 */
struct AuZn: public Metal {

    static constexpr const char* NAME = "AuZn";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};


} // namespace plask

#endif	//PLASK__AuZn_H
