#ifndef PLASK__GaInP_H
#define PLASK__GaInP_H

/** @file
This file includes undoped GaInP
*/

#include <plask/material/material.h>
#include "GaP.h"
#include "InP.h"

namespace plask {

/**
 * Represent undoped GaInP, its physical properties.
 */
struct GaInP: public Semiconductor {

    static constexpr const char* NAME = "GaInP";

    GaInP(const Material::Composition& Comp);
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
    virtual double A(double T) const;
    virtual double B(double T) const;
    virtual double C(double T) const;
    virtual double D(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    double Ga,
           In;

    GaP mGaP;
    InP mInP;

};

} // namespace plask

#endif	//PLASK__GaInP_H
