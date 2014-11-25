#include "GaPSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaPSb::GaPSb(const Material::Composition& Comp) {
    P = Comp.find("P")->second;
    Sb = Comp.find("Sb")->second;
}

std::string GaPSb::str() const { return StringBuilder("Ga")("P")("Sb", Sb); }

std::string GaPSb::name() const { return NAME; }

MI_PROPERTY(GaPSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb")
            )
double GaPSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = P*mGaP.lattC(T,'a') + Sb*mGaSb.lattC(T,'a');
    else if (x == 'c') tLattC = P*mGaP.lattC(T,'a') + Sb*mGaSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaPSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaP, GaSb")
            )
double GaPSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = P*mGaP.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == 'X') tEg = P*mGaP.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == 'L') tEg = P*mGaP.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - P*Sb*2.7;
    else if (point == '*')
    {
        double tEgG = P*mGaP.Eg(T,e,'G') + Sb*mGaSb.Eg(T,e,'G') - P*Sb*2.7;
        double tEgX = P*mGaP.Eg(T,e,'X') + Sb*mGaSb.Eg(T,e,'X') - P*Sb*2.7;
        double tEgL = P*mGaP.Eg(T,e,'L') + Sb*mGaSb.Eg(T,e,'L') - P*Sb*2.7;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaPSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::Dso(double T, double e) const {
    return ( P*mGaP.Dso(T, e) + Sb*mGaSb.Dso(T, e) );
}

MI_PROPERTY(GaPSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaPSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = P*mGaP.Me(T,e,point).c00 + Sb*mGaSb.Me(T,e,point).c00,
        tMe.c11 = P*mGaP.Me(T,e,point).c11 + Sb*mGaSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = P*mGaP.Me(T,e,point).c00 + Sb*mGaSb.Me(T,e,point).c00;
        tMe.c11 = P*mGaP.Me(T,e,point).c11 + Sb*mGaSb.Me(T,e,point).c11;
    }
    return ( tMe );
}

MI_PROPERTY(GaPSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaPSb::Mhh(double T, double e) const {
    double lMhh = P*mGaP.Mhh(T,e).c00 + Sb*mGaSb.Mhh(T,e).c00,
           vMhh = P*mGaP.Mhh(T,e).c11 + Sb*mGaSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaPSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaPSb::Mlh(double T, double e) const {
    double lMlh = P*mGaP.Mlh(T,e).c00 + Sb*mGaSb.Mlh(T,e).c00,
           vMlh = P*mGaP.Mlh(T,e).c11 + Sb*mGaSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaPSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaPSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaPSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaPSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaPSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::VB(double T, double e, char point, char hole) const {
    double tVB( P*mGaP.VB(T,0.,point,hole) + Sb*mGaSb.VB(T,0.,point,hole) );
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

MI_PROPERTY(GaPSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::ac(double T) const {
    return ( P*mGaP.ac(T) + Sb*mGaSb.ac(T) );
}

MI_PROPERTY(GaPSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::av(double T) const {
    return ( P*mGaP.av(T) + Sb*mGaSb.av(T) );
}

MI_PROPERTY(GaPSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::b(double T) const {
    return ( P*mGaP.b(T) + Sb*mGaSb.b(T) );
}

MI_PROPERTY(GaPSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::d(double T) const {
    return ( P*mGaP.d(T) + Sb*mGaSb.d(T) );
}

MI_PROPERTY(GaPSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::c11(double T) const {
    return ( P*mGaP.c11(T) + Sb*mGaSb.c11(T) );
}

MI_PROPERTY(GaPSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::c12(double T) const {
    return ( P*mGaP.c12(T) + Sb*mGaSb.c12(T) );
}

MI_PROPERTY(GaPSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::c44(double T) const {
    return ( P*mGaP.c44(T) + Sb*mGaSb.c44(T) );
}

MI_PROPERTY(GaPSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaP, GaSb")
            )
Tensor2<double> GaPSb::thermk(double T, double t) const {
    double lCondT = 1./(P/mGaP.thermk(T,t).c00 + Sb/mGaSb.thermk(T,t).c00 + P*Sb*0.16),
           vCondT = 1./(P/mGaP.thermk(T,t).c11 + Sb/mGaSb.thermk(T,t).c11 + P*Sb*0.16);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaPSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::dens(double T) const {
    return ( P*mGaP.dens(T) + Sb*mGaSb.dens(T) );
}

MI_PROPERTY(GaPSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaP, GaSb"),
            MIComment("no temperature dependence")
            )
double GaPSb::cp(double T) const {
    return ( P*mGaP.cp(T) + Sb*mGaSb.cp(T) );
}

MI_PROPERTY(GaPSb, nr,
            MIComment("TODO")
            )
double GaPSb::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for GaPSb");
}

MI_PROPERTY(GaPSb, absp,
            MIComment("TODO")
            )
double GaPSb::absp(double wl, double T) const {
    throw NotImplemented("absp for GaPSb");
}

bool GaPSb::isEqual(const Material &other) const {
    const GaPSb& o = static_cast<const GaPSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<GaPSb> materialDB_register_GaPSb;

}} // namespace plask::materials
