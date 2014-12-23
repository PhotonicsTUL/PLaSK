#include "GaAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

GaAsSb::GaAsSb(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string GaAsSb::str() const { return StringBuilder("Ga")("As")("Sb", Sb); }

std::string GaAsSb::name() const { return NAME; }

MI_PROPERTY(GaAsSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb")
            )
double GaAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mGaAs.lattC(T,'a') + Sb*mGaSb.lattC(T,'a');
    else if (x == 'c') tLattC = As*mGaAs.lattC(T,'a') + Sb*mGaSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(GaAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, GaSb")
            )
double GaAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mGaAs.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - As*Sb*1.43;
    else if (point == 'X') tEg = As*mGaAs.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - As*Sb*1.2;
    else if (point == 'L') tEg = As*mGaAs.Eg(T,e,point) + Sb*mGaSb.Eg(T,e,point) - As*Sb*1.2;
    else if (point == '*') {
        double tEgG = As*mGaAs.Eg(T,e,'G') + Sb*mGaSb.Eg(T,e,'G') - As*Sb*1.43;
        double tEgX = As*mGaAs.Eg(T,e,'X') + Sb*mGaSb.Eg(T,e,'X') - As*Sb*1.2;
        double tEgL = As*mGaAs.Eg(T,e,'L') + Sb*mGaSb.Eg(T,e,'L') - As*Sb*1.2;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaAsSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::Dso(double T, double e) const {
    return ( As*mGaAs.Dso(T, e) + Sb*mGaSb.Dso(T, e) - As*Sb*0.6 );
}

MI_PROPERTY(GaAsSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("nonlinear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAsSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = As*mGaAs.Me(T,e,point).c00 + Sb*mGaSb.Me(T,e,point).c00,
        tMe.c11 = As*mGaAs.Me(T,e,point).c11 + Sb*mGaSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = As*mGaAs.Me(T,e,point).c00 + Sb*mGaSb.Me(T,e,point).c00;
        tMe.c11 = As*mGaAs.Me(T,e,point).c11 + Sb*mGaSb.Me(T,e,point).c11;
    };
    if (point == 'G') {
        tMe.c00 += ( -As*Sb*(0.014) );
        tMe.c11 += ( -As*Sb*(0.014) );
    }
    return ( tMe );
}

MI_PROPERTY(GaAsSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAsSb::Mhh(double T, double e) const {
    double lMhh = As*mGaAs.Mhh(T,e).c00 + Sb*mGaSb.Mhh(T,e).c00,
           vMhh = As*mGaAs.Mhh(T,e).c11 + Sb*mGaSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(GaAsSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaAsSb::Mlh(double T, double e) const {
    double lMlh = As*mGaAs.Mlh(T,e).c00 + Sb*mGaSb.Mlh(T,e).c00,
           vMlh = As*mGaAs.Mlh(T,e).c11 + Sb*mGaSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(GaAsSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaAsSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaAsSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( As*mGaAs.VB(T,0.,point,hole) + Sb*mGaSb.VB(T,0.,point,hole) - As*Sb*(-1.06) );
    if (!e) return tVB;
    else {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
}

MI_PROPERTY(GaAsSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::ac(double T) const {
    return ( As*mGaAs.ac(T) + Sb*mGaSb.ac(T) );
}

MI_PROPERTY(GaAsSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::av(double T) const {
    return ( As*mGaAs.av(T) + Sb*mGaSb.av(T) );
}

MI_PROPERTY(GaAsSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::b(double T) const {
    return ( As*mGaAs.b(T) + Sb*mGaSb.b(T) );
}

MI_PROPERTY(GaAsSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::d(double T) const {
    return ( As*mGaAs.d(T) + Sb*mGaSb.d(T) );
}

MI_PROPERTY(GaAsSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::c11(double T) const {
    return ( As*mGaAs.c11(T) + Sb*mGaSb.c11(T) );
}

MI_PROPERTY(GaAsSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::c12(double T) const {
    return ( As*mGaAs.c12(T) + Sb*mGaSb.c12(T) );
}

MI_PROPERTY(GaAsSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::c44(double T) const {
    return ( As*mGaAs.c44(T) + Sb*mGaSb.c44(T) );
}

MI_PROPERTY(GaAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: GaAs, GaSb")
            )
Tensor2<double> GaAsSb::thermk(double T, double t) const {
    double lCondT = 1./(As/mGaAs.thermk(T,t).c00 + Sb/mGaSb.thermk(T,t).c00 + As*Sb*0.91),
           vCondT = 1./(As/mGaAs.thermk(T,t).c11 + Sb/mGaSb.thermk(T,t).c11 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(GaAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::dens(double T) const {
    return ( As*mGaAs.dens(T) + Sb*mGaSb.dens(T) );
}

MI_PROPERTY(GaAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: GaAs, GaSb"),
            MIComment("no temperature dependence")
            )
double GaAsSb::cp(double T) const {
    return ( As*mGaAs.cp(T) + Sb*mGaSb.cp(T) );
}

Material::ConductivityType GaAsSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaAsSb, nr,
            MIComment("TODO")
            )
double GaAsSb::nr(double wl, double T, double n) const {
    throw NotImplemented("nr for GaAsSb");
}

MI_PROPERTY(GaAsSb, absp,
            MIComment("TODO")
            )
double GaAsSb::absp(double wl, double T) const {
    throw NotImplemented("absp for GaAsSb");
}

bool GaAsSb::isEqual(const Material &other) const {
    const GaAsSb& o = static_cast<const GaAsSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<GaAsSb> materialDB_register_GaAsSb;

}} // namespace plask::materials
