#include "AlGaAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaAsSb::AlGaAsSb(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string AlGaAsSb::str() const { return StringBuilder("Al", Al)("Ga")("As")("Sb", Sb); }

std::string AlGaAsSb::name() const { return NAME; }

MI_PROPERTY(AlGaAsSb, lattC,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Ga*As*mGaAs.lattC(T,'a') + Ga*Sb*mGaSb.lattC(T,'a')
            + Al*As*mAlAs.lattC(T,'a') + Al*Sb*mAlSb.lattC(T,'a');
    else if (x == 'c') tLattC = Ga*As*mGaAs.lattC(T,'c') + Ga*Sb*mGaSb.lattC(T,'c')
            + Al*As*mAlAs.lattC(T,'c') + Al*Sb*mAlSb.lattC(T,'c');
    return ( tLattC );
}

MI_PROPERTY(AlGaAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Al*Ga*As*(-0.127+1.310*Al) - Al*Ga*Sb*(-0.044+1.22*Al) - Ga*As*Sb*(1.43) - Al*As*Sb*(0.8) - Al*Ga*As*Sb*0.48;
    else if (point == 'X') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Al*Ga*As*(0.055) - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
    else if (point == 'L') tEg = Ga*As*mGaAs.Eg(T,e,point) + Ga*Sb*mGaSb.Eg(T,e,point)
            + Al*As*mAlAs.Eg(T,e,point) + Al*Sb*mAlSb.Eg(T,e,point)
            - Ga*As*Sb*(1.2) - Al*As*Sb*(0.28);
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaAsSb, Dso,
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::Dso(double T, double e) const {
    return ( Ga*As*mGaAs.Dso(T,e) + Ga*Sb*mGaSb.Dso(T,e)
             + Al*As*mAlAs.Dso(T,e) + Al*Sb*mAlSb.Dso(T,e)
             - Al*Ga*Sb*(0.3) - Ga*As*Sb*(0.6) - Al*As*Sb*(0.15) );
}

MI_PROPERTY(AlGaAsSb, Me,
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
Tensor2<double> AlGaAsSb::Me(double T, double e, char point) const {
    double lMe = Ga*As*mGaAs.Me(T,e,point).c00 + Ga*Sb*mGaSb.Me(T,e,point).c00
            + Al*As*mAlAs.Me(T,e,point).c00 + Al*Sb*mAlSb.Me(T,e,point).c00
            - Ga*As*Sb*(0.014),
           vMe = Ga*As*mGaAs.Me(T,e,point).c11 + Ga*Sb*mGaSb.Me(T,e,point).c11
            + Al*As*mAlAs.Me(T,e,point).c11 + Al*Sb*mAlSb.Me(T,e,point).c11
            - Ga*As*Sb*(0.014);
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaAsSb, Mhh,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
Tensor2<double> AlGaAsSb::Mhh(double T, double e) const {
    double lMhh = Ga*As*mGaAs.Mhh(T,e).c00 + Ga*Sb*mGaSb.Mhh(T,e).c00
            + Al*As*mAlAs.Mhh(T,e).c00 + Al*Sb*mAlSb.Mhh(T,e).c00,
           vMhh = Ga*As*mGaAs.Mhh(T,e).c11 + Ga*Sb*mGaSb.Mhh(T,e).c11
            + Al*As*mAlAs.Mhh(T,e).c11 + Al*Sb*mAlSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaAsSb, Mlh,
            MISource("nonlinear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
Tensor2<double> AlGaAsSb::Mlh(double T, double e) const {
    double lMlh = Ga*As*mGaAs.Mlh(T,e).c00 + Ga*Sb*mGaSb.Mlh(T,e).c00
            + Al*As*mAlAs.Mlh(T,e).c00 + Al*Sb*mAlSb.Mlh(T,e).c00,
           vMlh = Ga*As*mGaAs.Mlh(T,e).c11 + Ga*Sb*mGaSb.Mlh(T,e).c11
            + Al*As*mAlAs.Mlh(T,e).c11 + Al*Sb*mAlSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaAsSb, VB,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( Ga*As*mGaAs.VB(T,0.,point,hole) + Ga*Sb*mGaSb.VB(T,0.,point,hole)
                + Al*As*mAlAs.VB(T,0.,point,hole) + Al*Sb*mAlSb.VB(T,0.,point,hole)
                - Al*As*Sb*(-1.71) - Ga*As*Sb*(-1.06) );
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

MI_PROPERTY(AlGaAsSb, ac,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::ac(double T) const {
    return ( Ga*As*mGaAs.ac(T) + Ga*Sb*mGaSb.ac(T)
             + Al*As*mAlAs.ac(T) + Al*Sb*mAlSb.ac(T) );
}

MI_PROPERTY(AlGaAsSb, av,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::av(double T) const {
    return ( Ga*As*mGaAs.av(T) + Ga*Sb*mGaSb.av(T) + Al*As*mAlAs.av(T) + Al*Sb*mAlSb.av(T) );
}

MI_PROPERTY(AlGaAsSb, b,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::b(double T) const {
    return ( Ga*As*mGaAs.b(T) + Ga*Sb*mGaSb.b(T) + Al*As*mAlAs.b(T) + Al*Sb*mAlSb.b(T) );
}

MI_PROPERTY(AlGaAsSb, d,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::d(double T) const {
    return ( Ga*As*mGaAs.d(T) + Ga*Sb*mGaSb.d(T) + Al*As*mAlAs.d(T) + Al*Sb*mAlSb.d(T) );
}

MI_PROPERTY(AlGaAsSb, c11,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::c11(double T) const {
    return ( Ga*As*mGaAs.c11(T) + Ga*Sb*mGaSb.c11(T) + Al*As*mAlAs.c11(T) + Al*Sb*mAlSb.c11(T) );
}

MI_PROPERTY(AlGaAsSb, c12,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::c12(double T) const {
    return ( Ga*As*mGaAs.c12(T) + Ga*Sb*mGaSb.c12(T) + Al*As*mAlAs.c12(T) + Al*Sb*mAlSb.c12(T) );
}

MI_PROPERTY(AlGaAsSb, c44,
            MISource("linear interpolation: GaSb, AlSb, GaAs, AlAs")
            )
double AlGaAsSb::c44(double T) const {
    return ( Ga*As*mGaAs.c44(T) + Ga*Sb*mGaSb.c44(T) + Al*As*mAlAs.c44(T) + Al*Sb*mAlSb.c44(T) );
}

MI_PROPERTY(AlGaAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlGaAsSb::thermk(double T, double t) const {
    double lCondT = 1./(Ga*As/mGaAs.thermk(T,t).c00 + Ga*Sb/mGaSb.thermk(T,t).c00
                        + Al*As/mAlAs.thermk(T,t).c00 + Al*Sb/mAlSb.thermk(T,t).c00
                        + Al*Ga*0.32 + As*Sb*0.91),
           vCondT = 1./(Ga*As/mGaAs.thermk(T,t).c11 + Ga*Sb/mGaSb.thermk(T,t).c11
                        + Al*As/mAlAs.thermk(T,t).c11 + Al*Sb/mAlSb.thermk(T,t).c11
                        + Al*Ga*0.32 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaAsSb, nr,
            MIComment("TODO")
            )
double AlGaAsSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlGaAsSb, absp,
            MIComment("TODO")
            )
double AlGaAsSb::absp(double wl, double T) const {
    return ( 0. );
}

bool AlGaAsSb::isEqual(const Material &other) const {
    const AlGaAsSb& o = static_cast<const AlGaAsSb&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaAsSb> materialDB_register_AlGaAsSb;

}} // namespace plask::materials
