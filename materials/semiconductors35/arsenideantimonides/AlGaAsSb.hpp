#ifndef PLASK__AlGaAsSb_H
#define PLASK__AlGaAsSb_H

/** @file
This file contains undoped AlGaAsSb
*/

#include "plask/material/material.hpp"
#include "../antimonides/AlSb.hpp"
#include "../antimonides/GaSb.hpp"
#include "../arsenides/AlAs.hpp"
#include "../arsenides/GaAs.hpp"

namespace plask { namespace materials {

/**
 * Represent undoped AlGaAsSb, its physical properties.
 */
struct AlGaAsSb: public Semiconductor {

    static constexpr const char* NAME = "AlGaAsSb";

    AlGaAsSb(const Material::Composition& Comp);
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
    double cp(double T) const override;
    ConductivityType condtype() const override;
    double nr(double lam, double T, double n = .0) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double Al,
           Ga,
           As,
           Sb;

    AlSb mAlSb;
    GaSb mGaSb;
    AlAs mAlAs;
    GaAs mGaAs;

};

}} // namespace plask::materials

#endif	//PLASK__AlGaAsSb_H
