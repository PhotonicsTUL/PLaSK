#include "AlInP.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlInP::AlInP(const Material::Composition& Comp) {
    Al = Comp.find("Al")->second;
    In = Comp.find("In")->second;
}

std::string AlInP::str() const { return StringBuilder("Al", Al)("In")("P"); }

std::string AlInP::name() const { return NAME; }

MI_PROPERTY(AlInP, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = Al*mAlP.lattC(T,'a') + In*mInP.lattC(T,'a');
    else if (x == 'c') tLattC = Al*mAlP.lattC(T,'a') + In*mInP.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(AlInP, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: AlP, InP")
            )
double AlInP::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = Al*mAlP.Eg(T, e, point) + In*mInP.Eg(T, e, point) - Al*In*(-0.48);
    else if (point == 'X') tEg = Al*mAlP.Eg(T, e, point) + In*mInP.Eg(T, e, point) - Al*In*0.38;
    else if (point == 'L') tEg = Al*mAlP.Eg(T, e, point) + In*mInP.Eg(T, e, point);
    else if (point == '*')
    {
        double tEgG = Al*mAlP.Eg(T, e, 'G') + In*mInP.Eg(T, e, 'G') - Al*In*(-0.48);
        double tEgX = Al*mAlP.Eg(T, e, 'X') + In*mInP.Eg(T, e, 'X') - Al*In*0.38;
        double tEgL = Al*mAlP.Eg(T, e, 'L') + In*mInP.Eg(T, e, 'L');
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlInP, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("nonlinear interpolation: AlP, InP")
            )
double AlInP::Dso(double T, double e) const {
    return ( Al*mAlP.Dso(T,e) + In*mInP.Dso(T,e) - Al*In*(-0.19) );
}

MI_PROPERTY(AlInP, Me,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, InP")
            )
Tensor2<double> AlInP::Me(double T, double e, char point) const {
    double lMe = Al*mAlP.Me(T,e,point).c00 + In*mInP.Me(T,e,point).c00,
           vMe = Al*mAlP.Me(T,e,point).c11 + In*mInP.Me(T,e,point).c11;
    return ( Tensor2<double>(lMe,vMe) );
}

MI_PROPERTY(AlInP, Mhh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, InP")
            )
Tensor2<double> AlInP::Mhh(double T, double e) const {
    double lMhh = Al*mAlP.Mhh(T,e).c00 + In*mInP.Mhh(T,e).c00,
           vMhh = Al*mAlP.Mhh(T,e).c11 + In*mInP.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlInP, Mlh,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, InP")
            )
Tensor2<double> AlInP::Mlh(double T, double e) const {
    double lMlh = Al*mAlP.Mlh(T,e).c00 + In*mInP.Mlh(T,e).c00,
           vMlh = Al*mAlP.Mlh(T,e).c11 + In*mInP.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlInP, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlInP::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlInP, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::VB(double T, double e, char point, char hole) const {
    double tVB( Al*mAlP.VB(T,0.,point,hole) + In*mInP.VB(T,0.,point,hole) );
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

MI_PROPERTY(AlInP, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::ac(double T) const {
    return ( Al*mAlP.ac(T) + In*mInP.ac(T) );
}

MI_PROPERTY(AlInP, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::av(double T) const {
    return ( Al*mAlP.av(T) + In*mInP.av(T) );
}

MI_PROPERTY(AlInP, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::b(double T) const {
    return ( Al*mAlP.b(T) + In*mInP.b(T) );
}

MI_PROPERTY(AlInP, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::d(double T) const {
    return ( Al*mAlP.d(T) + In*mInP.d(T) );
}

MI_PROPERTY(AlInP, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::c11(double T) const {
    return ( Al*mAlP.c11(T) + In*mInP.c11(T) );
}

MI_PROPERTY(AlInP, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::c12(double T) const {
    return ( Al*mAlP.c12(T) + In*mInP.c12(T) );
}

MI_PROPERTY(AlInP, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MISource("linear interpolation: AlP, InP")
            )
double AlInP::c44(double T) const {
    return ( Al*mAlP.c44(T) + In*mInP.c44(T) );
}

MI_PROPERTY(AlInP, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, Wiley 2009"),
            MISource("inversion od nonlinear interpolation of resistivity: AlP, InP")
            )
Tensor2<double> AlInP::thermk(double T, double t) const {
    double lCondT = 1./(Al/mAlP.thermk(T,t).c00 + In/mInP.thermk(T,t).c00 + Al*In*0.15),
           vCondT = 1./(Al/mAlP.thermk(T,t).c11 + In/mInP.thermk(T,t).c11 + Al*In*0.15);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlInP, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, InP"),
            MIComment("no temperature dependence")
            )
double AlInP::dens(double T) const {
    return ( Al*mAlP.dens(T) + In*mInP.dens(T) );
}

MI_PROPERTY(AlInP, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons, 2009"),
            MISource("linear interpolation: AlP, InP"),
            MIComment("no temperature dependence")
            )
double AlInP::cp(double T) const {
    return ( Al*mAlP.cp(T) + In*mInP.cp(T) );
}

MI_PROPERTY(AlInP, nr,
            MIComment("TODO")
            )
double AlInP::nr(double wl, double T, double n) const {
    return ( 0. );
}

MI_PROPERTY(AlInP, absp,
            MIComment("TODO")
            )
double AlInP::absp(double wl, double T) const {
    return ( 0. );
}

bool AlInP::isEqual(const Material &other) const {
    const AlInP& o = static_cast<const AlInP&>(other);
    return o.Al == this->Al;
}

static MaterialsDB::Register<AlInP> materialDB_register_AlInP;

}} // namespace plask::materials
