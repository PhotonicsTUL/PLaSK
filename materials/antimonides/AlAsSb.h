#ifndef PLASK__AlAsSb_H
#define PLASK__AlAsSb_H

/** @file
This file contains undoped AlAsSb
*/

#include <plask/material/material.h>
#include "../arsenides/AlAs.h"
#include "AlSb.h"

namespace plask { namespace materials {

/**
 * Represent undoped AlAsSb, its physical properties.
 */
struct AlAsSb: public Semiconductor {

    static constexpr const char* NAME = "AlAsSb";

    AlAsSb(const Material::Composition& Comp);
    virtual std::string str() const;
    virtual std::string name() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, double e, char point) const;
    virtual double Dso(double T, double e) const;
    virtual Tensor2<double> Me(double T, double e, char point) const;
    virtual Tensor2<double> Mhh(double T, double e) const;
    virtual Tensor2<double> Mlh(double T, double e) const;
    virtual double CB(double T, double e, char point) const;
    virtual double VB(double T, double e, char point, char hole) const;
    virtual double ac(double T) const;
    virtual double av(double T) const;
    virtual double b(double T) const;
    virtual double d(double T) const;
    virtual double c11(double T) const;
    virtual double c12(double T) const;
    virtual double c44(double T) const;
    virtual Tensor2<double> thermk(double T, double t) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;

protected:
    virtual bool isEqual(const Material& other) const;

protected:
    double As,
           Sb;

    AlAs mAlAs;
    AlSb mAlSb;

};

}} // namespace plask::materials

#endif	//PLASK__AlAsSb_H
