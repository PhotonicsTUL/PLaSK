#ifndef PLASK__AlAs_H
#define PLASK__AlAs_H

/** @file
This file contains undoped AlAs
*/

#include <plask/material/material.h>

namespace plask { namespace materials {

/**
 * Represent undoped AlAs, its physical properties.
 */
struct AlAs: public Semiconductor {

    static constexpr const char* NAME = "AlAs";

    virtual std::string name() const { return NAME; }

    virtual double lattC(double T, char x) const override {
        double tLattC(0.);
        if (x == 'a') tLattC = 5.6611 + 2.90e-5 * (T-300.);
        return tLattC;
    }

    virtual double Eg(double T, double e, char point) const override {
        double tEg(0.);
        if (point == 'G') tEg = phys::Varshni(3.099, 0.885e-3, 530., T);
        else if (point == 'X') tEg = phys::Varshni(2.24, 0.70e-3, 530., T);
        else if (point == 'L') tEg = phys::Varshni(2.46, 0.605e-3, 204., T);
        return tEg;
    }

    virtual double Dso(double T, double e) const override {
        return 0.28;
    }

    virtual Tensor2<double> Me(double T, double e, char point) const override {
        Tensor2<double> tMe(0., 0.);
        if (point == 'G') {
            tMe.c00 = 0.124;
            tMe.c11 = 0.124;
        }
        return tMe;
    }

    virtual Tensor2<double> Mhh(double T, double e) const override {
        Tensor2<double> tMhh(0.51, 0.51); // [001]
        return tMhh;
    }

    virtual Tensor2<double> Mlh(double T, double e) const override {
        Tensor2<double> tMlh(0.18, 0.18);
        return tMlh;
    }

    virtual double CB(double T, double e, char point) const override {
        double tCB( VB(T,0.,point, 'H') + Eg(T,0.,point) );
        if (!e) return tCB;
        else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
    }

    virtual double VB(double T, double e, char point, char hole) const override {
        double tVB(-1.33);
        if (e) {
            double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
            double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
            if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
            else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
            else return 0.;
        }
        return tVB;
    }

    virtual double ac(double T) const override {
        return -5.64;
    }

    virtual double av(double T) const override {
        return 2.47;
    }

    virtual double b(double T) const override {
        return -2.3;
    }

    virtual double d(double T) const override {
        return -3.4;
    }

    virtual double c11(double T) const override {
        return 125.0;
    }

    virtual double c12(double T) const override {
        return 53.4;
    }

    virtual double c44(double T) const override {
        return 54.2;
    }

    virtual Tensor2<double> thermk(double T, double t) const override {
        double tk = 91. * pow((300./T),1.375);
        return(Tensor2<double>(tk, tk));
    }

    virtual double nr(double wl, double T, double n = .0) const override {
        double L2 = wl*wl*1e-6;
        double nR296K = sqrt(1.+7.055*L2/(L2-0.068));
        return ( nR296K + nR296K*4.6e-5*(T-296.) );
    }

    double eps(double T) const override {
        return 10.1;
    }
    
  protected:
      
    virtual bool isEqual(const Material &other) const override {
        return true;
    }
    
};

}} // namespace plask::materials

#endif	//PLASK__AlAs_H
