#ifndef PLASK__AlInAs_H
#define PLASK__AlInAs_H

/** @file
This file contains undoped AlInAs
*/

#include <plask/material/material.h>
#include "AlAs.h"
#include "InAs.h"

namespace plask { namespace materials {

/**
 * Represent undoped AlInAs, its physical properties.
 */
struct AlInAs: public Semiconductor {

    static constexpr const char* NAME = "AlInAs";

    AlInAs(const Material::Composition& Comp);
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
    double Al,
           In;

    AlAs mAlAs;
    InAs mInAs;

};

}} // namespace plask::materials

#endif	//PLASK__AlInAs_H
