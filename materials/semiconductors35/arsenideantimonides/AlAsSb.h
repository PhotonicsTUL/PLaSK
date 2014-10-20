#ifndef PLASK__AlAsSb_H
#define PLASK__AlAsSb_H

/** @file
This file contains undoped AlAsSb
*/

#include <plask/material/material.h>
#include "../arsenides/AlAs.h"
#include "../antimonides/AlSb.h"

namespace plask { namespace materials {

/**
 * Represent undoped AlAsSb, its physical properties.
 */
struct AlAsSb: public Semiconductor {

    static constexpr const char* NAME = "AlAsSb";

    AlAsSb(const Material::Composition& Comp);
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
    double As,
           Sb;

    AlAs mAlAs;
    AlSb mAlSb;

};

}} // namespace plask::materials

#endif	//PLASK__AlAsSb_H
