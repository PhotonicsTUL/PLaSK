#ifndef PLASK__AlAsSb_H
#define PLASK__AlAsSb_H

/** @file
This file contains undoped AlAsSb
*/

#include "plask/material/material.hpp"
#include "../arsenides/AlAs.hpp"
#include "../antimonides/AlSb.hpp"

namespace plask { namespace materials {

/**
 * Represent undoped AlAsSb, its physical properties.
 */
struct AlAsSb: public Semiconductor {

    static constexpr const char* NAME = "AlAsSb";

    AlAsSb(const Material::Composition& Comp);
    std::string str() const override;
    std::string name() const override;
    Composition composition() const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e, char point) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    Tensor2<double> Mh(double T, double e) const override;
    double CB(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double ac(double T) const override;
    double av(double T) const override;
    double b(double T) const override;
    double d(double T) const override;
    double c11(double T) const override;
    double c12(double T) const override;
    double c44(double T) const override;
    Tensor2<double> thermk(double T, double t) const override;
    double dens(double T) const override;
    ConductivityType condtype() const override;
    double cp(double T) const override;
    double nr(double lam, double T, double n = .0) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double As,
           Sb;

    AlAs mAlAs;
    AlSb mAlSb;

};

}} // namespace plask::materials

#endif	//PLASK__AlAsSb_H