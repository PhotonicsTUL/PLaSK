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

    std::string name() const override;
    Kind kind() const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e, char point) const override;
    double CB(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    double Mso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    Tensor2<double> Mh(double T, double e) const override;
    double eps(double T) const override;
    double chi(double T, double e, char point) const override;
    double Ni(double T) const override;
    double Nf(double T) const override;
    double EactD(double T) const override;
    double EactA(double T) const override;
    Tensor2<double> mob(double T) const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double A(double T) const override;
    double B(double T) const override;
    double C(double T) const override;
    double D(double T) const override;
    Tensor2<double> thermk(double T, double h) const override;
    double dens(double T) const override;
    double cp(double T) const override;
    double nr(double wl, double T, double n = .0) const override;
    double absp(double wl, double T) const override;
protected:
    bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif	//PLASK__AIR_H
