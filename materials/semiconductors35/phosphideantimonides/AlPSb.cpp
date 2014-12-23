#include "AlPSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlPSb::AlPSb(const Material::Composition& Comp) {
    P = Comp.find("P")->second;
    Sb = Comp.find("Sb")->second;
}

std::string AlPSb::str() const { return StringBuilder("Al")("P")("Sb", Sb); }

std::string AlPSb::name() const { return NAME; }

MI_PROPERTY(AlPSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb")
            )
double AlPSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = P*mAlP.lattC(T,'a') + Sb*mAlSb.lattC(T,'a');
    else if (x == 'c') tLattC = P*mAlP.lattC(T,'a') + Sb*mAlSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(AlPSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlP, AlSb")
            )
double AlPSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = P*mAlP.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == 'X') tEg = P*mAlP.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == 'L') tEg = P*mAlP.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == '*') {
        double tEgG = P*mAlP.Eg(T,e,'G') + Sb*mAlSb.Eg(T,e,'G') - P*Sb*2.7;
        double tEgX = P*mAlP.Eg(T,e,'X') + Sb*mAlSb.Eg(T,e,'X') - P*Sb*2.7;
        double tEgL = P*mAlP.Eg(T,e,'L') + Sb*mAlSb.Eg(T,e,'L') - P*Sb*2.7;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlPSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::Dso(double T, double e) const {
    return ( P*mAlP.Dso(T, e) + Sb*mAlSb.Dso(T, e) );
}

MI_PROPERTY(AlPSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlPSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = P*mAlP.Me(T,e,point).c00 + Sb*mAlSb.Me(T,e,point).c00,
        tMe.c11 = P*mAlP.Me(T,e,point).c11 + Sb*mAlSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = P*mAlP.Me(T,e,point).c00 + Sb*mAlSb.Me(T,e,point).c00;
        tMe.c11 = P*mAlP.Me(T,e,point).c11 + Sb*mAlSb.Me(T,e,point).c11;
    }
    return ( tMe );
}

MI_PROPERTY(AlPSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlPSb::Mhh(double T, double e) const {
    double lMhh = P*mAlP.Mhh(T,e).c00 + Sb*mAlSb.Mhh(T,e).c00,
           vMhh = P*mAlP.Mhh(T,e).c11 + Sb*mAlSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlPSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlPSb::Mlh(double T, double e) const {
    double lMlh = P*mAlP.Mlh(T,e).c00 + Sb*mAlSb.Mlh(T,e).c00,
           vMlh = P*mAlP.Mlh(T,e).c11 + Sb*mAlSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlPSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlPSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlPSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlPSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlPSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::VB(double T, double e, char point, char hole) const {
    double tVB( P*mAlP.VB(T,0.,point,hole) + Sb*mAlSb.VB(T,0.,point,hole) );
    if (!e) return tVB;
    else
    {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(AlPSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::ac(double T) const {
    return ( P*mAlP.ac(T) + Sb*mAlSb.ac(T) );
}

MI_PROPERTY(AlPSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::av(double T) const {
    return ( P*mAlP.av(T) + Sb*mAlSb.av(T) );
}

MI_PROPERTY(AlPSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::b(double T) const {
    return ( P*mAlP.b(T) + Sb*mAlSb.b(T) );
}

MI_PROPERTY(AlPSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::d(double T) const {
    return ( P*mAlP.d(T) + Sb*mAlSb.d(T) );
}

MI_PROPERTY(AlPSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::c11(double T) const {
    return ( P*mAlP.c11(T) + Sb*mAlSb.c11(T) );
}

MI_PROPERTY(AlPSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::c12(double T) const {
    return ( P*mAlP.c12(T) + Sb*mAlSb.c12(T) );
}

MI_PROPERTY(AlPSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::c44(double T) const {
    return ( P*mAlP.c44(T) + Sb*mAlSb.c44(T) );
}

MI_PROPERTY(AlPSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: AlP, AlSb")
            )
Tensor2<double> AlPSb::thermk(double T, double t) const {
    double lCondT = 1./(P/mAlP.thermk(T,t).c00 + Sb/mAlSb.thermk(T,t).c00 + P*Sb*0.16),
           vCondT = 1./(P/mAlP.thermk(T,t).c11 + Sb/mAlSb.thermk(T,t).c11 + P*Sb*0.16);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlPSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::dens(double T) const {
    return ( P*mAlP.dens(T) + Sb*mAlSb.dens(T) );
}

MI_PROPERTY(AlPSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: AlP, AlSb"),
            MIComment("no temperature dependence")
            )
double AlPSb::cp(double T) const {
    return ( P*mAlP.cp(T) + Sb*mAlSb.cp(T) );
}

Material::ConductivityType AlPSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlPSb, nr,
            MIComment("TODO")
            )
double AlPSb::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for AlPSb");
}

MI_PROPERTY(AlPSb, absp,
            MIComment("TODO")
            )
double AlPSb::absp(double wl, double T) const {
    throw NotImplemented("absp for AlPSb");
}

bool AlPSb::isEqual(const Material &other) const {
    const AlPSb& o = static_cast<const AlPSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<AlPSb> materialDB_register_AlPSb;

}} // namespace plask::materials
