#include "InAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

InAsSb::InAsSb(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string InAsSb::str() const { return StringBuilder("In")("As")("Sb", Sb); }

std::string InAsSb::name() const { return NAME; }

MI_PROPERTY(InAsSb, lattC,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mInAs.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    else if (x == 'c') tLattC = As*mInAs.lattC(T,'a') + Sb*mInSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(InAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.67;
    else if (point == 'X') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.6;
    else if (point == 'L') tEg = As*mInAs.Eg(T,e,point) + Sb*mInSb.Eg(T,e,point) - As*Sb*0.6;
    else if (point == '*')
    {
        double tEgG = As*mInAs.Eg(T,e,'G') + Sb*mInSb.Eg(T,e,'G') - As*Sb*0.67;
        double tEgX = As*mInAs.Eg(T,e,'X') + Sb*mInSb.Eg(T,e,'X') - As*Sb*0.6;
        double tEgL = As*mInAs.Eg(T,e,'L') + Sb*mInSb.Eg(T,e,'L') - As*Sb*0.6;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(InAsSb, Dso,
            MISource("nonlinear interpolation: InAs, InSb")
            )
double InAsSb::Dso(double T, double e) const {
    return ( As*mInAs.Dso(T, e) + Sb*mInSb.Dso(T, e) - As*Sb*1.2 );
}

MI_PROPERTY(InAsSb, Me,
            MISource("nonlinear interpolation: InAs, InSb")
            )
Tensor2<double> InAsSb::Me(double T, double e, char point) const {
    double lMe = As*mInAs.Me(T,e,point).c00 + Sb*mInSb.Me(T,e,point).c00 - As*Sb*0.027,
           vMe = As*mInAs.Me(T,e,point).c11 + Sb*mInSb.Me(T,e,point).c11 - As*Sb*0.027;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(InAsSb, Mhh,
            MISource("linear interpolation: InAs, InSb")
            )
Tensor2<double> InAsSb::Mhh(double T, double e) const {
    double lMhh = As*mInAs.Mhh(T,e).c00 + Sb*mInSb.Mhh(T,e).c00,
           vMhh = As*mInAs.Mhh(T,e).c11 + Sb*mInSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(InAsSb, Mlh,
            MISource("linear interpolation: InAs, InSb")
            )
Tensor2<double> InAsSb::Mlh(double T, double e) const {
    double lMlh = As*mInAs.Mlh(T,e).c00 + Sb*mInSb.Mlh(T,e).c00,
           vMlh = As*mInAs.Mlh(T,e).c11 + Sb*mInSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(InAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double InAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(InAsSb, VB,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( As*mInAs.VB(T,0.,point,hole) + Sb*mInSb.VB(T,0.,point,hole) );
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

MI_PROPERTY(InAsSb, ac,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::ac(double T) const {
    return ( As*mInAs.ac(T) + Sb*mInSb.ac(T) );
}

MI_PROPERTY(InAsSb, av,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::av(double T) const {
    return ( As*mInAs.av(T) + Sb*mInSb.av(T) );
}

MI_PROPERTY(InAsSb, b,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::b(double T) const {
    return ( As*mInAs.b(T) + Sb*mInSb.b(T) );
}

MI_PROPERTY(InAsSb, d,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::d(double T) const {
    return ( As*mInAs.d(T) + Sb*mInSb.d(T) );
}

MI_PROPERTY(InAsSb, c11,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::c11(double T) const {
    return ( As*mInAs.c11(T) + Sb*mInSb.c11(T) );
}

MI_PROPERTY(InAsSb, c12,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::c12(double T) const {
    return ( As*mInAs.c12(T) + Sb*mInSb.c12(T) );
}

MI_PROPERTY(InAsSb, c44,
            MISource("linear interpolation: InAs, InSb")
            )
double InAsSb::c44(double T) const {
    return ( As*mInAs.c44(T) + Sb*mInSb.c44(T) );
}

MI_PROPERTY(InAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> InAsSb::thermk(double T, double t) const {
    double lCondT = 1./(As/mInAs.thermk(T,t).c00 + Sb/mInSb.thermk(T,t).c00 + As*Sb*0.91),
           vCondT = 1./(As/mInAs.thermk(T,t).c11 + Sb/mInSb.thermk(T,t).c11 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(InAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::dens(double T) const {
    return ( As*mInAs.dens(T) + Sb*mInSb.dens(T) );
}

MI_PROPERTY(InAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: InAs, InSb"),
            MIComment("no temperature dependence")
            )
double InAsSb::cp(double T) const {
    return ( As*mInAs.cp(T) + Sb*mInSb.cp(T) );
}

MI_PROPERTY(InAsSb, nr,
            MIComment("TODO")
            )
double InAsSb::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(InAsSb, absp,
            MIComment("TODO")
            )
double InAsSb::absp(double wl, double T) const {
    return ( 0. );
}

bool InAsSb::isEqual(const Material &other) const {
    const InAsSb& o = static_cast<const InAsSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<InAsSb> materialDB_register_InAsSb;

}} // namespace plask::materials
