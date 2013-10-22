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

    virtual std::string name() const { return NAME; }

    virtual double lattC(double T, char x) const {
        double tLattC(0.);
        if (x == 'a') tLattC = 5.65325 + 3.88e-5 * (T-300.);
        return tLattC;
    }

    virtual double Eg(double T, double e, char point) const {
        double tEg(0.);
        if (point == 'G') tEg = phys::Varshni(1.519, 0.5405e-3, 204., T);
        else if (point == 'X') tEg = phys::Varshni(1.981, 0.460e-3, 204., T);
        else if (point == 'L') tEg = phys::Varshni(1.815, 0.605e-3, 204., T);
        return tEg;
    }

    virtual double Dso(double T, double e) const {
        return 0.341;
    }

    virtual Tensor2<double> Me(double T, double e, char point) const {
        Tensor2<double> tMe(0., 0.);
        if (point == 'G') {
            tMe.c00 = 0.067;
            tMe.c11 = 0.067;
        }
        return tMe;
    }

    virtual Tensor2<double> Mhh(double T, double e) const {
        Tensor2<double> tMhh(0.33, 0.33); // [001]
        return tMhh;
    }

    virtual Tensor2<double> Mlh(double T, double e) const {
        Tensor2<double> tMlh(0.090, 0.090);
        return tMlh;
    }

    virtual double CB(double T, double e, char point) const {
        double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
        if (!e) return tCB;
        else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
    }

    virtual double VB(double T, double e, char point, char hole) const {
        double tVB(-0.80);
        if (!e) return tVB;
        else return tVB + 2.*av(T)*(1.-c12(T)/c11(T))*e;
    }

    virtual double ac(double T) const {
        return -7.17;
    }

    virtual double av(double T) const {
        return 1.16;
    }

    virtual double b(double T) const {
        return -2.0;
    }

    virtual double d(double T) const {
        return -4.8;
    }

    virtual double c11(double T) const {
        return 122.1;
    }

    virtual double c12(double T) const {
        return 56.6;
    }

    virtual double c44(double T) const {
        return 60.0;
    }

    virtual Tensor2<double> thermk(double T, double t) const {
        double tCondT = 45.*pow((300./T),1.25);
        return Tensor2<double>(tCondT, tCondT);
    }

    virtual Tensor2<double> cond(double T) const {
        double c = 1e2 * phys::qe * 8000.* pow((300./T), 2./3.) * 1e16;
        return Tensor2<double>(c, c);
    }

    virtual double nr(double wl, double T) const {
        return 0.;
    }

    virtual double absp(double wl, double T) const {
        return 0.;
    }

  protected:
    virtual bool isEqual(const Material &other) const {
        return true;
    }
};

}} // namespace plask::materials

#endif	//PLASK__GaAs_H
