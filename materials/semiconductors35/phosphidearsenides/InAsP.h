#ifndef PLASK__InAsP_H
#define PLASK__InAsP_H

/** @file
This file contains undoped InAsP
*/

#include <plask/material/material.h>
#include "../arsenides/InAs.h"
#include "../phosphides/InP.h"

namespace plask { namespace materials {

/**
 * Represent undoped InAsP, its physical properties.
 */
struct InAsP: public Semiconductor {

    static constexpr const char* NAME = "InAsP";

    InAsP(const Material::Composition& Comp);
    virtual std::string str() const override;
    virtual std::string name() const override;
    virtual double lattC(double T, char x) const override;
    virtual double Eg(double T, double e, char point) const override;
    virtual double Dso(double T, double e) const override;
    virtual Tensor2<double> Me(double T, double e, char point) const override;
    virtual Tensor2<double> Mhh(double T, double e) const override;
    virtual Tensor2<double> Mlh(double T, double e) const override;
    virtual Tensor2<double> Mh(double T, double e) const override;
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
    virtual ConductivityType condtype() const override;
    virtual double nr(double lam, double T, double n = .0) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

protected:
    double As,
           P;

    InAs mInAs;
    InP mInP;

};

}} // namespace plask::materials

#endif	//PLASK__InAsP_H
