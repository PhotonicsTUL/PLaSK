#ifndef PLASK__AuGe_H
#define PLASK__AuGe_H

/** @file
This file includes AuGe
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent AuGe, its physical properties.
 */
struct AuGe: public Metal {

    static constexpr const char* NAME = "AuGe";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};


} // namespace plask

#endif	//PLASK__AuGe_H
