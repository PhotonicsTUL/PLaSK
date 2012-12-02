#ifndef PLASK__GaInAs_H
#define PLASK__GaInAs_H

/** @file
This file includes undoped GaInAs
*/

#include <plask/material/material.h>
#include "GaAs.h"
#include "InAs.h"

namespace plask {

/**
 * Represent undoped GaInAs, its physical properties.
 */
struct GaInAs: public Semiconductor {

    static constexpr const char* NAME = "GaInAs";

    GaInAs(const Material::Composition& Comp);
    virtual std::string str() const;
    virtual std::string name() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, char point) const;
    virtual double Dso(double T) const;
    virtual Tensor2<double> Me(double T, char point) const;
    virtual Tensor2<double> Mhh(double T, char point) const;
    virtual Tensor2<double> Mlh(double T, char point) const;
    virtual double ac(double T) const;
    virtual double av(double T) const;
    virtual double b(double T) const;
    virtual double c11(double T) const;
    virtual double c12(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    double Ga,
           In;

    GaAs mGaAs;
    InAs mInAs;

};

} // namespace plask

#endif	//PLASK__GaInAs_H
