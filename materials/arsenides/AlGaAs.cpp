#include "AlGaAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaAs::AlGaAs(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

std::string AlGaAs::str() const { return StringBuilder("Al", Al)("Ga")("As"); }

std::string AlGaAs::name() const { return NAME; }

MI_PROPERTY(AlGaAs, lattC,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlAs.lattC(T,'a') + Ga*mGaAs.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlAs.lattC(T,'c') + Ga*mGaAs.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) - Al*Ga*(-0.127+1.310*Al);
    else if (point == 'X') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point) - Al*Ga*(0.055);
    else if (point == 'L') tEg = Al*mAlAs.Eg(T,e,point) + Ga*mGaAs.Eg(T,e,point);
    return ( tEg );
}

MI_PROPERTY(AlGaAs, Dso,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::Dso(double T, double e) const {
    return ( Al*mAlAs.Dso(T,e) + Ga*mGaAs.Dso(T,e) );
}

MI_PROPERTY(AlGaAs, Me,
            MISource("linear interpolation: AlAs, GaAs")
            )
Tensor2<double> AlGaAs::Me(double T, double e, char point) const {
    double lMe = Al*mAlAs.Me(T,e,point).c00 + Ga*mGaAs.Me(T,e,point).c00,
           vMe = Al*mAlAs.Me(T,e,point).c11 + Ga*mGaAs.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaAs, Mhh,
            MISource("linear interpolation: AlAs, GaAs")
            )
Tensor2<double> AlGaAs::Mhh(double T, double e) const {
    double lMhh = Al*mAlAs.Mhh(T,e).c00 + Ga*mGaAs.Mhh(T,e).c00,
           vMhh = Al*mAlAs.Mhh(T,e).c11 + Ga*mGaAs.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaAs, Mlh,
            MISource("linear interpolation: AlAs, GaAs")
            )
Tensor2<double> AlGaAs::Mlh(double T, double e) const {
    double lMlh = Al*mAlAs.Mlh(T,e).c00 + Ga*mGaAs.Mlh(T,e).c00,
           vMlh = Al*mAlAs.Mlh(T,e).c11 + Ga*mGaAs.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaAs, VB,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlAs.VB(T,e,point,hole) + Ga*mGaAs.VB(T,e,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
}

MI_PROPERTY(AlGaAs, ac,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::ac(double T) const {
    return ( Al*mAlAs.ac(T) + Ga*mGaAs.ac(T) );
}

MI_PROPERTY(AlGaAs, av,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::av(double T) const {
    return ( Al*mAlAs.av(T) + Ga*mGaAs.av(T) );
}

MI_PROPERTY(AlGaAs, b,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::b(double T) const {
    return ( Al*mAlAs.b(T) + Ga*mGaAs.b(T) );
}

MI_PROPERTY(AlGaAs, d,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::d(double T) const {
    return ( Al*mAlAs.d(T) + Ga*mGaAs.d(T) );
}

MI_PROPERTY(AlGaAs, c11,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::c11(double T) const {
    return ( Al*mAlAs.c11(T) + Ga*mGaAs.c11(T) );
}

MI_PROPERTY(AlGaAs, c12,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::c12(double T) const {
    return ( Al*mAlAs.c12(T) + Ga*mGaAs.c12(T) );
}

MI_PROPERTY(AlGaAs, c44,
            MISource("linear interpolation: AlAs, GaAs")
            )
double AlGaAs::c44(double T) const {
    return ( Al*mAlAs.c44(T) + Ga*mGaAs.c44(T) );
}

MI_PROPERTY(AlGaAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlGaAs::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlAs.thermk(T,t).c00 + Ga/mGaAs.thermk(T,t).c00 + Al*Ga*0.32),
           vCondT = 1./(Al/mAlAs.thermk(T,t).c11 + Ga/mGaAs.thermk(T,t).c11 + Al*Ga*0.32);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAs, nr,
            MISource("D.T.F. Marple, J. Appl. Phys. 35 (1964) 1241-1242"),
            MISource("S. Gehrsitz, J. Appl. Phys. 87 (2000) 7825-7837"),
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIComment("fit by Lukasz Piskorski")
            )
double AlGaAs::nr(double wl, double T) const {
    double L2 = wl*wl*1e-6;
    double nR296K = sqrt(1.+(9.659-2.604*Al)*L2/(L2-(0.137-0.069*Al)));
    return ( nR296K + nR296K*(Al*4.6e-5+Ga*4.5e-5)*(T-296.) );
}

MI_PROPERTY(AlGaAs, absp,
            MIComment("TODO")
            )
double AlGaAs::absp(double wl, double T) const {
    return ( 0. );
}

bool AlGaAs::isEqual(const Material &other) const {
    const AlGaAs& o = static_cast<const AlGaAs&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaAs> materialDB_register_AlGaAs;

}} // namespace plask::materials
