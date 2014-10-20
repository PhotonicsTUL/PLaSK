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
    virtual std::string str() const override;
    virtual std::string name() const override;
    virtual double lattC(double T, char x) const override;
    virtual double Eg(double T, double e, char point) const override;
    virtual double Dso(double T, double e) const override;
    virtual Tensor2<double> Me(double T, double e, char point) const override;
    virtual Tensor2<double> Mhh(double T, double e) const override;
    virtual Tensor2<double> Mlh(double T, double e) const override;
    virtual double CB(double T, double e, char point) const override;
    virtual double VB(double T, double e, char point, char hole) const override;
    virtual double ac(double T) const override;
    virtual double av(double T) const override;
    virtual double b(double T) const override;
    virtual double d(double T) const override;
    virtual double c11(double T) const override;
    virtual double c12(double T) const override;
    virtual double c44(double T) const override;
    virtual Tensor2<double> thermk(double T, double t) const override;
    virtual double dens(double T) const override;
    virtual double cp(double T) const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

protected:
    double Al,
           In;

    AlAs mAlAs;
    InAs mInAs;

};

}} // namespace plask::materials

#endif	//PLASK__AlInAs_H
