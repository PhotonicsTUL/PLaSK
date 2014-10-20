#include "AlGaP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlGaP::AlGaP(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    Ga = Comp.find("Ga")->second;
}

std::string AlGaP::str() const { return StringBuilder("Al", Al)("Ga")("P"); }

std::string AlGaP::name() const { return NAME; }

MI_PROPERTY(AlGaP, lattC,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlP.lattC(T,'a') + Ga*mGaP.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlP.lattC(T,'a') + Ga*mGaP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(AlGaP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlP.Eg(T, e, point) + Ga*mGaP.Eg(T, e, point);
    else if (point == 'X') tEg = Al*mAlP.Eg(T, e, point) + Ga*mGaP.Eg(T, e, point) - Al*Ga*0.13;
    else if (point == 'L') tEg = Al*mAlP.Eg(T, e, point) + Ga*mGaP.Eg(T, e, point);
    else if (point == '*')
    {
        double tEgG = Al*mAlP.Eg(T, e, 'G') + Ga*mGaP.Eg(T, e, 'G');
        double tEgX = Al*mAlP.Eg(T, e, 'X') + Ga*mGaP.Eg(T, e, 'X') - Al*Ga*0.13;
        double tEgL = Al*mAlP.Eg(T, e, 'L') + Ga*mGaP.Eg(T, e, 'L');
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlGaP, Dso,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::Dso(double T, double e) const {
    return ( Al*mAlP.Dso(T,e) + Ga*mGaP.Dso(T,e) );
}

MI_PROPERTY(AlGaP, Me,
            MISource("linear interpolation: AlP, GaP")
            )
Tensor2<double> AlGaP::Me(double T, double e, char point) const {
    double lMe = Al*mAlP.Me(T,e,point).c00 + Ga*mGaP.Me(T,e,point).c00,
           vMe = Al*mAlP.Me(T,e,point).c11 + Ga*mGaP.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlGaP, Mhh,
            MISource("linear interpolation: AlP, GaP")
            )
Tensor2<double> AlGaP::Mhh(double T, double e) const {
    double lMhh = Al*mAlP.Mhh(T,e).c00 + Ga*mGaP.Mhh(T,e).c00,
           vMhh = Al*mAlP.Mhh(T,e).c11 + Ga*mGaP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlGaP, Mlh,
            MISource("linear interpolation: AlP, GaP")
            )
Tensor2<double> AlGaP::Mlh(double T, double e) const {
    double lMlh = Al*mAlP.Mlh(T,e).c00 + Ga*mGaP.Mlh(T,e).c00,
           vMlh = Al*mAlP.Mlh(T,e).c11 + Ga*mGaP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlGaP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlGaP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlGaP, VB,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlP.VB(T,0.,point,hole) + Ga*mGaP.VB(T,0.,point,hole) );
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

MI_PROPERTY(AlGaP, ac,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::ac(double T) const {
    return ( Al*mAlP.ac(T) + Ga*mGaP.ac(T) );
}

MI_PROPERTY(AlGaP, av,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::av(double T) const {
    return ( Al*mAlP.av(T) + Ga*mGaP.av(T) );
}

MI_PROPERTY(AlGaP, b,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::b(double T) const {
    return ( Al*mAlP.b(T) + Ga*mGaP.b(T) );
}

MI_PROPERTY(AlGaP, d,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::d(double T) const {
    return ( Al*mAlP.d(T) + Ga*mGaP.d(T) );
}

MI_PROPERTY(AlGaP, c11,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::c11(double T) const {
    return ( Al*mAlP.c11(T) + Ga*mGaP.c11(T) );
}

MI_PROPERTY(AlGaP, c12,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::c12(double T) const {
    return ( Al*mAlP.c12(T) + Ga*mGaP.c12(T) );
}

MI_PROPERTY(AlGaP, c44,
            MISource("linear interpolation: AlP, GaP")
            )
double AlGaP::c44(double T) const {
    return ( Al*mAlP.c44(T) + Ga*mGaP.c44(T) );
}

MI_PROPERTY(AlGaP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009")
            )
Tensor2<double> AlGaP::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlP.thermk(T,t).c00 + Ga/mGaP.thermk(T,t).c00 + Al*Ga*0.32),
           vCondT = 1./(Al/mAlP.thermk(T,t).c11 + Ga/mGaP.thermk(T,t).c11 + Al*Ga*0.32);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlGaP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, GaP"),
            MIComment("no temperature dependence")
            )
double AlGaP::dens(double T) const {
    return ( Al*mAlP.dens(T) + Ga*mGaP.dens(T) );
}

MI_PROPERTY(AlGaP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, GaP"),
            MIComment("no temperature dependence")
            )
double AlGaP::cp(double T) const {
    return ( Al*mAlP.cp(T) + Ga*mGaP.cp(T) );
}

MI_PROPERTY(AlGaP, nr,
            MIComment("TODO")
            )
double AlGaP::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlGaP, absp,
            MIComment("TODO")
            )
double AlGaP::absp(double wl, double T) const {
    return ( 0. );
}

bool AlGaP::isEqual(const Material &other) const {
    const AlGaP& o = static_cast<const AlGaP&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlGaP> materialDB_register_AlGaP;

}} // namespace plask::materials
