#ifndef PLASK__GaAs_H
#define PLASK__GaAs_H

/** @file
This file contains undoped GaAs
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped GaAs, its physical properties.
 */
struct GaAs: public Semiconductor {

    static constexpr const char* NAME = "GaAs";

    std::string name() const override;

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
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double dens(double T) const override;
    double cp(double T) const override;
    double nr(double wl, double T, double n = .0) const override;
    double absp(double wl, double T) const override;
    double eps(double T) const override;

  protected:

    virtual bool isEqual(const Material &other) const override;};

}} // namespace plask::materials

#endif	//PLASK__GaAs_H
