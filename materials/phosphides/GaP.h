#ifndef PLASK__GaP_H
#define PLASK__GaP_H

/** @file
This file includes undoped GaP
*/

#include <plask/material/material.h>

namespace plask {

/**
 * Represent undoped GaP, its physical properties.
 */
struct GaP: public Semiconductor {

    static constexpr const char* NAME = "GaP";

    virtual std::string name() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual double Dso(double T) const;
    virtual std::pair<double,double> Me(double T, char point) const;
    virtual std::pair<double,double> Mhh(double T, char point) const;
    virtual std::pair<double,double> Mlh(double T, char point) const;
    virtual double ac(double T) const;
    virtual double av(double T) const;
    virtual double b(double T) const;
    virtual double c11(double T) const;
    virtual double c12(double T) const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

};

} // namespace plask

#endif	//PLASK__GaP_H
