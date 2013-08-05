#ifndef PLASK__AIR_H
#define PLASK__AIR_H

/** @file
This file contains undoped AlN
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped AlN, its physical properties.
 */
struct Air: public Material {

    static constexpr const char* NAME = "air";

    virtual std::string name() const;
    virtual Kind kind() const;
    virtual double lattC(double T, char x) const;
    virtual double Eg(double T, double e, char point) const;
    virtual double CBO(double T, double e, char point) const;
    virtual double VBO(double T, double e, char point) const;
    virtual double Dso(double T, double e) const;
    virtual double Mso(double T, double e) const;
    virtual Tensor2<double> Me(double T, double e, char point) const;
    virtual Tensor2<double> Mhh(double T, double e) const;
    virtual Tensor2<double> Mlh(double T, double e) const;
    virtual Tensor2<double> Mh(double T, double e) const;
    virtual double eps(double T) const;
    virtual double chi(double T, double e, char point) const;
    virtual double Nc(double T, double e, char point) const;
    virtual double Nc(double T, double e) const;
    virtual double Ni(double T) const;
    virtual double Nf(double T) const;
    virtual double EactD(double T) const;
    virtual double EactA(double T) const;
    virtual Tensor2<double> mob(double T) const;
    virtual Tensor2<double> cond(double T) const;
    virtual ConductivityType condtype() const;
    virtual double A(double T) const;
    virtual double B(double T) const;
    virtual double C(double T) const;
    virtual double D(double T) const;
    virtual Tensor2<double> thermk(double T, double h) const;
    virtual double dens(double T) const;
    virtual double cp(double T) const;
    virtual double nr(double wl, double T) const;
    virtual double absp(double wl, double T) const;
protected:
    virtual bool isEqual(const Material& other) const;
};


}} // namespace plask::materials

#endif	//PLASK__AIR_H
