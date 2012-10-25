#ifndef PLASK__Ti_H
#define PLASK__Ti_H

/** @file
This file includes Ti
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent Ti, its physical properties.
 */
struct Ti: public Metal {

    static constexpr const char* NAME = "Ti";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};


} // namespace plask

#endif	//PLASK__Ti_H
