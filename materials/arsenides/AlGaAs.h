#ifndef PLASK__AlGaAs_H
#define PLASK__AlGaAs_H

/** @file
This file includes undoped AlGaAs
*/

#include <plask/material/material.h>
#include "GaAs.h"
#include "AlAs.h"

namespace plask {

/**
 * Represent undoped AlGaAs, its physical properties.
 */
struct AlGaAs: public Semiconductor {

    static constexpr const char* NAME = "AlGaAs";

    AlGaAs(const Material::Composition& Comp);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual std::pair<double,double> thermCond(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double Eg(double T, char point) const;
    virtual double lattC(double T, char x) const;

protected:
    double Al,
           Ga;

    GaAs mGaAs;
    AlAs mAlAs;

};

} // namespace plask

#endif	//PLASK__AlGaAs_H
