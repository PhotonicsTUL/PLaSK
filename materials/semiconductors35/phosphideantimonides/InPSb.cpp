#include "InPSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

InPSb::InPSb(const Material::Composition& Comp) {
    P = Comp.find("P")->second;
    Sb = Comp.find("Sb")->second;
}

std::string InPSb::str() const { return StringBuilder("In")("P")("Sb", Sb); }

std::string InPSb::name() const { return NAME; }

MI_PROPERTY(InPSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = P*mInP.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = P*mInP.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(InPSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: InP, InSb")
            )
double InPSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = P*mInP.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - P*Sb*1.9;
    else if (point == 'X') tEg = P*mInP.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - P*Sb*1.9;
    else if (point == 'L') tEg = P*mInP.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - P*Sb*1.9;
    else if (point == '*')
    {
        double tEgG = P*mInP.Eg(T,e,'G') + Sb*mInSb.Eg(T,e,'G') - P*Sb*1.9;
        double tEgX = P*mInP.Eg(T,e,'X') + Sb*mInSb.Eg(T,e,'X') - P*Sb*1.9;
        double tEgL = P*mInP.Eg(T,e,'L') + Sb*mInSb.Eg(T,e,'L') - P*Sb*1.9;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InPSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: InP, InSb")
            )
double InPSb::Dso(double T, double e) const {
    return ( P*mInP.Dso(T, e) + Sb*mInSb.Dso(T, e) - P*Sb*0.75 );
}

MI_PROPERTY(InPSb, Me,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InP, InSb")
            )
Tensor2<double> InPSb::Me(double T, double e, char point) const {
    double lMe = P*mInP.Me(T,e,point).c00 + Sb*mInSb.Me(T,e,point).c00,
           vMe = P*mInP.Me(T,e,point).c11 + Sb*mInSb.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(InPSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InP, InSb")
            )
Tensor2<double> InPSb::Mhh(double T, double e) const {
    double lMhh = P*mInP.Mhh(T,e).c00 + Sb*mInSb.Mhh(T,e).c00,
           vMhh = P*mInP.Mhh(T,e).c11 + Sb*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(InPSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InP, InSb")
            )
Tensor2<double> InPSb::Mlh(double T, double e) const {
    double lMlh = P*mInP.Mlh(T,e).c00 + Sb*mInSb.Mlh(T,e).c00,
           vMlh = P*mInP.Mlh(T,e).c11 + Sb*mInSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(InPSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InPSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InPSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::VB(double T, double e, char point, char hole) const {
    double tVB( P*mInP.VB(T,0.,point,hole) + Sb*mInSb.VB(T,0.,point,hole) );
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

MI_PROPERTY(InPSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::ac(double T) const {
    return ( P*mInP.ac(T) + Sb*mInSb.ac(T) );
}

MI_PROPERTY(InPSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::av(double T) const {
    return ( P*mInP.av(T) + Sb*mInSb.av(T) );
}

MI_PROPERTY(InPSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::b(double T) const {
    return ( P*mInP.b(T) + Sb*mInSb.b(T) );
}

MI_PROPERTY(InPSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::d(double T) const {
    return ( P*mInP.d(T) + Sb*mInSb.d(T) );
}

MI_PROPERTY(InPSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::c11(double T) const {
    return ( P*mInP.c11(T) + Sb*mInSb.c11(T) );
}

MI_PROPERTY(InPSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::c12(double T) const {
    return ( P*mInP.c12(T) + Sb*mInSb.c12(T) );
}

MI_PROPERTY(InPSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: InP, InSb")
            )
double InPSb::c44(double T) const {
    return ( P*mInP.c44(T) + Sb*mInSb.c44(T) );
}

MI_PROPERTY(InPSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MISource("inversion od nonlinear interpolation of resistivity: InP, InSb")
            )
Tensor2<double> InPSb::thermk(double T, double t) const {
    double lCondT = 1./(P/mInP.thermk(T,t).c00 + Sb/mInSb.thermk(T,t).c00 + P*Sb*0.16),
           vCondT = 1./(P/mInP.thermk(T,t).c11 + Sb/mInSb.thermk(T,t).c11 + P*Sb*0.16);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(InPSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InP, InSb"),
            MIComment("no temperature dependence")
            )
double InPSb::dens(double T) const {
    return ( P*mInP.dens(T) + Sb*mInSb.dens(T) );
}

MI_PROPERTY(InPSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InP, InSb"),
            MIComment("no temperature dependence")
            )
double InPSb::cp(double T) const {
    return ( P*mInP.cp(T) + Sb*mInSb.cp(T) );
}

MI_PROPERTY(InPSb, nr,
            MIComment("TODO")
            )
double InPSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(InPSb, absp,
            MIComment("TODO")
            )
double InPSb::absp(double wl, double T) const {
    return ( 0. );
}

bool InPSb::isEqual(const Material &other) const {
    const InPSb& o = static_cast<const InPSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<InPSb> materialDB_register_InPSb;

}} // namespace plask::materials
