#include "AlAsSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

AlAsSb::AlAsSb(const Material::Composition& Comp) {
    As = Comp.find("As")->second;
    Sb = Comp.find("Sb")->second;
}

std::string AlAsSb::str() const { return StringBuilder("Al")("As")("Sb", Sb); }

std::string AlAsSb::name() const { return NAME; }

MI_PROPERTY(AlAsSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb")
            )
double AlAsSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = As*mAlAs.lattC(T,'a') + Sb*mAlSb.lattC(T,'a');
    else if (x == 'c') tLattC = As*mAlAs.lattC(T,'a') + Sb*mAlSb.lattC(T,'a');
    return ( tLattC );
}

MI_PROPERTY(AlAsSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, AlSb")
            )
double AlAsSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = As*mAlAs.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - As*Sb*0.8;
    else if (point == 'X') tEg = As*mAlAs.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - As*Sb*0.28;
    else if (point == 'L') tEg = As*mAlAs.Eg(T,e,point) + Sb*mAlSb.Eg(T,e,point) - As*Sb*0.28;
    else if (point == '*') {
        double tEgG = As*mAlAs.Eg(T,e,'G') + Sb*mAlSb.Eg(T,e,'G') - As*Sb*0.8;
        double tEgX = As*mAlAs.Eg(T,e,'X') + Sb*mAlSb.Eg(T,e,'X') - As*Sb*0.28;
        double tEgL = As*mAlAs.Eg(T,e,'L') + Sb*mAlSb.Eg(T,e,'L') - As*Sb*0.28;
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(AlAsSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::Dso(double T, double e) const {
    return ( As*mAlAs.Dso(T, e) + Sb*mAlSb.Dso(T, e) - As*Sb*0.15 );
}

MI_PROPERTY(AlAsSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAsSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    if ((point == 'G') || (point == 'X') || (point == 'L')) {
        tMe.c00 = As*mAlAs.Me(T,e,point).c00 + Sb*mAlSb.Me(T,e,point).c00,
        tMe.c11 = As*mAlAs.Me(T,e,point).c11 + Sb*mAlSb.Me(T,e,point).c11;
    }
    else if (point == '*') {
        point = 'G';
        if ( Eg(T,e,'X') == Eg(T,e,'*') ) point = 'X';
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) point = 'L';
        tMe.c00 = As*mAlAs.Me(T,e,point).c00 + Sb*mAlSb.Me(T,e,point).c00;
        tMe.c11 = As*mAlAs.Me(T,e,point).c11 + Sb*mAlSb.Me(T,e,point).c11;
    }
    return ( tMe );
}

MI_PROPERTY(AlAsSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAsSb::Mhh(double T, double e) const {
    double lMhh = As*mAlAs.Mhh(T,e).c00 + Sb*mAlSb.Mhh(T,e).c00,
           vMhh = As*mAlAs.Mhh(T,e).c11 + Sb*mAlSb.Mhh(T,e).c11;
    return ( Tensor2<double>(lMhh,vMhh) );
}

MI_PROPERTY(AlAsSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
Tensor2<double> AlAsSb::Mlh(double T, double e) const {
    double lMlh = As*mAlAs.Mlh(T,e).c00 + Sb*mAlSb.Mlh(T,e).c00,
           vMlh = As*mAlAs.Mlh(T,e).c11 + Sb*mAlSb.Mlh(T,e).c11;
    return ( Tensor2<double>(lMlh,vMlh) );
}

MI_PROPERTY(AlAsSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlAsSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlAsSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double AlAsSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point,'H') + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(AlAsSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("nonlinear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::VB(double T, double e, char point, char hole) const {
    double tVB( As*mAlAs.VB(T,0.,point,hole) + Sb*mAlSb.VB(T,0.,point,hole) - As*Sb*(-1.71) );
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

MI_PROPERTY(AlAsSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::ac(double T) const {
    return ( As*mAlAs.ac(T) + Sb*mAlSb.ac(T) );
}

MI_PROPERTY(AlAsSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::av(double T) const {
    return ( As*mAlAs.av(T) + Sb*mAlSb.av(T) );
}

MI_PROPERTY(AlAsSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::b(double T) const {
    return ( As*mAlAs.b(T) + Sb*mAlSb.b(T) );
}

MI_PROPERTY(AlAsSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::d(double T) const {
    return ( As*mAlAs.d(T) + Sb*mAlSb.d(T) );
}

MI_PROPERTY(AlAsSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::c11(double T) const {
    return ( As*mAlAs.c11(T) + Sb*mAlSb.c11(T) );
}

MI_PROPERTY(AlAsSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::c12(double T) const {
    return ( As*mAlAs.c12(T) + Sb*mAlSb.c12(T) );
}

MI_PROPERTY(AlAsSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::c44(double T) const {
    return ( As*mAlAs.c44(T) + Sb*mAlSb.c44(T) );
}

MI_PROPERTY(AlAsSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37; "), // temperature dependence for binaries
            MISource("inversion of nonlinear interpolation of resistivity: AlAs, AlSb")
            )
Tensor2<double> AlAsSb::thermk(double T, double t) const {
    double lCondT = 1./(As/mAlAs.thermk(T,t).c00 + Sb/mAlSb.thermk(T,t).c00 + As*Sb*0.91),
           vCondT = 1./(As/mAlAs.thermk(T,t).c11 + Sb/mAlSb.thermk(T,t).c11 + As*Sb*0.91);
    return ( Tensor2<double>(lCondT,vCondT) );
}

MI_PROPERTY(AlAsSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::dens(double T) const {
    return ( As*mAlAs.dens(T) + Sb*mAlSb.dens(T) );
}

MI_PROPERTY(AlAsSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52; "),
            MISource("linear interpolation: AlAs, AlSb"),
            MIComment("no temperature dependence")
            )
double AlAsSb::cp(double T) const {
    return ( As*mAlAs.cp(T) + Sb*mAlSb.cp(T) );
}

Material::ConductivityType AlAsSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(AlAsSb, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::wl, 500, 7000) // TODO
            )
double AlAsSb::nr(double wl, double T, double n) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double tE0 = 3.2;
    double tEd = 28.;
    double tEG = 2.338;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO

    double nR = nR300K; // TODO // for E << Eg: dnR/dn = 0
    double dnRdT = As*4.6e-5 + Sb*1.19e-5; // from Adachi (2005) ebook p.243 tab. 10.6
    return ( nR + nR*dnRdT*(T-300.) );
}

MI_PROPERTY(AlAsSb, absp,
            MIComment("TODO")
            )
double AlAsSb::absp(double wl, double T) const {
    throw NotImplemented("absp for AlAsSb");
}

bool AlAsSb::isEqual(const Material &other) const {
    const AlAsSb& o = static_cast<const AlAsSb&>(other);
    return o.Sb == this->Sb;
}

static MaterialsDB::Register<AlAsSb> materialDB_register_AlAsSb;

}} // namespace plask::materials
