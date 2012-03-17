#ifndef PLASK__GaN_H
#define PLASK__GaN_H

/** @file
This file includes undoped GaN
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped GaN, its physical properties.
 */
struct GaN: public Semiconductor {

    static constexpr const char* NAME = "GaN";

    virtual std::string name() const;
    virtual double cond(double T) const;
    virtual double condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
};


} // namespace plask

#endif	//PLASK__GaN_H
