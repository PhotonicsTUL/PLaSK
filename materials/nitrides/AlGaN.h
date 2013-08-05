#ifndef PLASK__AlGaN_H
#define PLASK__AlGaN_H

/** @file
This file contains undoped AlGaN
*/

#include <plask/material/material.h>
#include "GaN.h"
#include "AlN.h"

namespace plask { namespace materials {

/**
 * Represent undoped AlGaN, its physical properties.
 */
struct AlGaN: public Semiconductor {

    static constexpr const char* NAME = "AlGaN";

    AlGaN(const Material::Composition& Comp);
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
    double Al,
           Ga;

    GaN mGaN;
    AlN mAlN;

};


}} // namespace plask::materials

#endif	//PLASK__AlGaN_H
