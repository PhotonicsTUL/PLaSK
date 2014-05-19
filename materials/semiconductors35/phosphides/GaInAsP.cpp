#include "GaInAsP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaInAsP::GaInAsP(const Material::Composition& Comp) {
    Ga = Comp.find("Ga")->second;
    In = Comp.find("In")->second;
    As = Comp.find("As")->second;
    P = Comp.find("P")->second;
}

std::string GaInAsP::str() const { return StringBuilder("Ga")("In", In)("As")("P", P); }

std::string GaInAsP::name() const { return NAME; }

MI_PROPERTY(GaInAsP, lattC,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*P*mGaP.lattC(T,'a')
            + In*As*mInAs.lattC(T,'a') + In*P*mInP.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*P*mGaP.lattC(T,'c')
            + In*As*mInAs.lattC(T,'c') + In*P*mInP.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(GaInAsP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAsP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*P*mGaP.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*P*mInP.Eg(T,e,point)
            - Ga*In*As*(0.477) - Ga*In*P*(0.65) - Ga*As*P*(0.19) - In*As*P*(0.10) - Ga*In*As*P*(0.13);
    else if (point == 'X') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*P*mGaP.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*P*mInP.Eg(T,e,point)
            - Ga*In*As*(1.4) - Ga*In*P*(0.20) - Ga*As*P*(0.24) - In*As*P*(0.27);
    else if (point == 'L') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*P*mGaP.Eg(T,e,point)
            + In*As*mInAs.Eg(T,e,point) + In*P*mInP.Eg(T,e,point)
            - Ga*In*As*(0.33) - Ga*In*P*(1.03) - Ga*As*P*(0.16) - In*As*P*(0.27);
    return ( tEg );
}

MI_PROPERTY(GaInAsP, Dso,
            MISource("nonlinear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*P*mGaP.Dso(T,e)
             + In*As*mInAs.Dso(T,e) + In*P*mInP.Dso(T,e)
             - Ga*In*As*(0.15) /*- Ga*In*P*(0.) - Ga*As*P*(0.)*/ - In*As*P*(0.16) );
}

MI_PROPERTY(GaInAsP, Me,
            MISource("nonlinear interpolation: GaP, InP, GaAs, InAs")
            )
Tensor2<double> GaInAsP::Me(double T, double e, char point) const {
    double lMe = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*P*mGaP.Me(T,e,point).c00
            + In*As*mInAs.Me(T,e,point).c00 + In*P*mInP.Me(T,e,point).c00
            - Ga*In*As*(0.008) - Ga*In*P*(0.01854) /*- Ga*As*P*(0.) - In*As*P*(0.)*/,
           vMe = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*P*mGaP.Me(T,e,point).c11
            + In*As*mInAs.Me(T,e,point).c11 + In*P*mInP.Me(T,e,point).c11
            - Ga*In*As*(0.008) - Ga*In*P*(0.01854) /*- Ga*As*P*(0.) - In*As*P*(0.)*/;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(GaInAsP, Mhh,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
Tensor2<double> GaInAsP::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*P*mGaP.Mhh(T,e).c00
            + In*As*mInAs.Mhh(T,e).c00 + In*P*mInP.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*P*mGaP.Mhh(T,e).c11
            + In*As*mInAs.Mhh(T,e).c11 + In*P*mInP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaInAsP, Mlh,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
Tensor2<double> GaInAsP::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*P*mGaP.Mlh(T,e).c00
            + In*As*mInAs.Mlh(T,e).c00 + In*P*mInP.Mlh(T,e).c00,
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*P*mGaP.Mlh(T,e).c11
            + In*As*mInAs.Mlh(T,e).c11 + In*P*mInP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaInAsP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaInAsP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaInAsP, VB,
            MISource("nonlinear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*As*mGaAs.VB(T,0.,point,hole) + Ga*P*mGaP.VB(T,0.,point,hole)
                + In*As*mInAs.VB(T,0.,point,hole) + In*P*mInP.VB(T,0.,point,hole)
                - Ga*In*As*(-0.38) /*- Ga*In*P*(0.) - Ga*As*P*(0.) - In*As*P*(0.)*/ );
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

MI_PROPERTY(GaInAsP, ac,
            MISource("nonlinear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*P*mGaP.ac(T)
             + In*As*mInAs.ac(T) + In*P*mInP.ac(T)
             - Ga*In*As*(2.61) /*- Ga*In*P*(0.) - Ga*As*P*(0.) - In*As*P*(0.)*/ );
}

MI_PROPERTY(GaInAsP, av,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*P*mGaP.av(T) + In*As*mInAs.av(T) + In*P*mInP.av(T) );
}

MI_PROPERTY(GaInAsP, b,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*P*mGaP.b(T) + In*As*mInAs.b(T) + In*P*mInP.b(T) );
}

MI_PROPERTY(GaInAsP, d,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*P*mGaP.d(T) + In*As*mInAs.d(T) + In*P*mInP.d(T) );
}

MI_PROPERTY(GaInAsP, c11,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*P*mGaP.c11(T) + In*As*mInAs.c11(T) + In*P*mInP.c11(T) );
}

MI_PROPERTY(GaInAsP, c12,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*P*mGaP.c12(T) + In*As*mInAs.c12(T) + In*P*mInP.c12(T) );
}

MI_PROPERTY(GaInAsP, c44,
            MISource("linear interpolation: GaP, InP, GaAs, InAs")
            )
double GaInAsP::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*P*mGaP.c44(T) + In*As*mInAs.c44(T) + In*P*mInP.c44(T) );
}

MI_PROPERTY(GaInAsP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> GaInAsP::thermk(double T, double t) const {
    double lCondT = 1./(Ga*As/mGaAs.thermk(T,t).c00 + Ga*P/mGaP.thermk(T,t).c00
                        + In*As/mInAs.thermk(T,t).c00 + In*P/mInP.thermk(T,t).c00
                        + Ga*In*0.72 + As*P*0.25),
           vCondT = 1./(Ga*As/mGaAs.thermk(T,t).c11 + Ga*P/mGaP.thermk(T,t).c11
                        + In*As/mInAs.thermk(T,t).c11 + In*P/mInP.thermk(T,t).c11
                        + Ga*In*0.72 + As*P*0.25);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaInAsP, nr,
            MIComment("TODO")
            )
double GaInAsP::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(GaInAsP, absp,
            MIComment("TODO")
            )
double GaInAsP::absp(double wl, double T) const {
    return ( 0. );
}

bool GaInAsP::isEqual(const Material &other) const {
    const GaInAsP& o = static_cast<const GaInAsP&>(other);
    return o.In == this->In;
}

static MaterialsDB::Register<GaInAsP> materialDB_register_GaInAsP;

}} // namespace plask::materials
