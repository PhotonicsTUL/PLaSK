#ifndef PLASK__GaAs_H
#define PLASK__GaAs_H

/** @file
This file includes undoped GaAs
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped GaAs, its physical properties.
 */
struct GaAs: public Semiconductor {

    static constexpr const char* NAME = "GaAs";

    virtual std::string name() const;
    virtual std::pair<double,double> cond(double T) const;
    virtual std::pair<double,double> condT(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;

};


} // namespace plask

#endif	//PLASK__GaAs_H
