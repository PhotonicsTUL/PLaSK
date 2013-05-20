#ifndef PLASK__InGaN_H
#define PLASK__InGaN_H

/** @file
This file contains undoped InGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "InN.h"

namespace plask {

/**
 * Represent undoped InGaN, its physical properties.
 */
struct InGaN: public Semiconductor {

    static constexpr const char* NAME = "InGaN";

    InGaN(const Material::Composition& Comp);
    virtual std::string name() const;
    virtual std::string str() const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
    virtual double Eg(double T, double e, char point) const;
    virtual Tensor2<double> Me(double T, double e, char point) const;
    virtual Tensor2<double> Mhh(double T, double e) const;
    virtual Tensor2<double> Mlh(double T, double e) const;
    virtual double lattC(double T, char x) const;

protected:
    virtual bool isEqual(const Material& other) const;

protected:
    double In,
           Ga;

    GaN mGaN;
    InN mInN;

};


} // namespace plask

#endif	//PLASK__InGaN_H
