#ifndef PLASK__AIR_H
#define PLASK__AIR_H

/** @file
This file contains air
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent air, its physical properties.
 */
struct PLASK_API Air: public Material {

    static constexpr const char* NAME = "air";

    virtual std::string name() const override;
    virtual Kind kind() const override;
    virtual double lattC(double T, char x) const override;
    virtual double Eg(double T, double e, char point) const override;
    virtual double CB(double T, double e, char point) const override;
    virtual double VB(double T, double e, char point, char hole) const override;
    virtual double Dso(double T, double e) const override;
    virtual double Mso(double T, double e) const override;
    virtual Tensor2<double> Me(double T, double e, char point) const override;
    virtual Tensor2<double> Mhh(double T, double e) const override;
    virtual Tensor2<double> Mlh(double T, double e) const override;
    virtual Tensor2<double> Mh(double T, double e) const override;
    virtual double eps(double T) const override;
    virtual double chi(double T, double e, char point) const override;
    virtual double Nc(double T, double e, char point) const override;
    virtual double Nv(double T, double e, char point) const override;
    virtual double Ni(double T) const override;
    virtual double Nf(double T) const override;
    virtual double EactD(double T) const override;
    virtual double EactA(double T) const override;
    virtual Tensor2<double> mob(double T) const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual ConductivityType condtype() const override;
    virtual double A(double T) const override;
    virtual double B(double T) const override;
    virtual double C(double T) const override;
    virtual double D(double T) const override;
    virtual Tensor2<double> thermk(double T, double h) const override;
    virtual double dens(double T) const override;
    virtual double cp(double T) const override;
    virtual double nr(double wl, double T, double n = .0) const override;
    virtual double absp(double wl, double T) const override;
protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__AIR_H
