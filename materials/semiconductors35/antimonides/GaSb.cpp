#include "GaSb.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb::name() const { return NAME; }

MI_PROPERTY(GaSb, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 6.0959 + 4.72e-5 * (T-300.);
    return ( tLattC );
}

MI_PROPERTY(GaSb, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(0.812, 0.417e-3, 140., T);
    else if (point == 'X') tEg = phys::Varshni(1.141, 0.475e-3, 94., T);
    else if (point == 'L') tEg = phys::Varshni(0.875, 0.597e-3, 140., T);
    else if (point == '*') {
        double tEgG = phys::Varshni(0.812, 0.417e-3, 140., T);
        double tEgX = phys::Varshni(1.141, 0.475e-3, 94., T);
        double tEgL = phys::Varshni(0.875, 0.597e-3, 140., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}

MI_PROPERTY(GaSb, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::Dso(double T, double e) const {
    return ( 0.76 );
}

MI_PROPERTY(GaSb, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.039), tMeX(1.08), tMeL(0.54);
    if (point == 'G') {
        tMe.c00 = tMeG; tMe.c11 = tMeG;
    }
    else if (point == 'X') {
        tMe.c00 = tMeX; tMe.c11 = tMeX;
    }
    else if (point == 'L') {
        tMe.c00 = tMeL; tMe.c11 = tMeL;
    }
    else if (point == '*') {
        if ( Eg(T,e,'G') == Eg(T,e,'*') ) {
            tMe.c00 = tMeG; tMe.c11 = tMeG;
        }
        else if ( Eg(T,e,'X') == Eg(T,e,'*') ) {
            tMe.c00 = tMeX; tMe.c11 = tMeX;
        }
        else if ( Eg(T,e,'L') == Eg(T,e,'*') ) {
            tMe.c00 = tMeL; tMe.c11 = tMeL;
        }
    }
    return ( tMe );
}

MI_PROPERTY(GaSb, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.22, 0.22); // [001]
    return ( tMhh );
}

MI_PROPERTY(GaSb, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaSb::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.045, 0.045); // [001]
    return ( tMlh );
}

MI_PROPERTY(GaSb, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> GaSb::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(GaSb, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
            )
double GaSb::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point) + Eg(T,0.,point) );
    if (!e) return ( tCB );
    else return ( tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e );
}

MI_PROPERTY(GaSb, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::VB(double T, double e, char point, char hole) const {
    double tVB(-0.03);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else throw NotImplemented("VB can be calculated only for holes: H, L");
    }
    return tVB;
}

MI_PROPERTY(GaSb, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::ac(double T) const {
    return ( -7.5 );
}

MI_PROPERTY(GaSb, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::av(double T) const {
    return ( 0.8 );
}

MI_PROPERTY(GaSb, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::b(double T) const {
    return ( -2.0 );
}

MI_PROPERTY(GaSb, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::d(double T) const {
    return ( -4.7 );
}

MI_PROPERTY(GaSb, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c11(double T) const {
    return ( 88.42 );
}

MI_PROPERTY(GaSb, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c12(double T) const {
    return ( 40.26 );
}

MI_PROPERTY(GaSb, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
            )
double GaSb::c44(double T) const {
    return ( 43.22 );
}

MI_PROPERTY(GaSb, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.37"), // temperature dependence
            MIArgumentRange(MaterialInfo::T, 50, 920)
           )
Tensor2<double> GaSb::thermk(double T, double t) const {
    double tCondT = 36.*pow((300./T),1.35);
    return ( Tensor2<double>(tCondT, tCondT) );
}

MI_PROPERTY(GaSb, cond,
            MISource("V. Sestakova et al., Proceedings of SPIE 4412 (2001) 161-165; "), // condRT = 1515 S/m
            MISource("N.K. Udayashankar et al., Bull. Mater. Sci. 24 (2001) 445-453; "), // condRT = 1923 S/m
            MISource("M.W. Heller et al., J. Appl. Phys. 57 (1985) 4626-4632"), // cond(T)
            MIComment("RT value: average value of electrical conductivity; "),
            MIComment("cond(T) = cond(300K)*(300/T)^d; d=0.53 assumed by L. Piskorski (PLaSK developer), based on Fig.1 from Heller")
            )
Tensor2<double> GaSb::cond(double T) const {
    double condT = 1700.*pow(300./T,0.53);
    return ( Tensor2<double>(condT, condT) );
}

MI_PROPERTY(GaSb, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double GaSb::dens(double T) const { return 5.61461e3; }

MI_PROPERTY(GaSb, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double GaSb::cp(double T) const { return 0.344e3; }

Material::ConductivityType GaSb::condtype() const { return Material::CONDUCTIVITY_I; }

MI_PROPERTY(GaSb, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264; "), // nR @ RT
            MISource("D.E. Aspnes et al., Phys. Rev. B 27 (1983) 985-1009; "), // nR @ RT
            MISource("S. Adachi, J. Appl. Phys. 66 (1989) 6030-6040; "), // nR @ RT
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2005) p.242"), // dnR/dT
            MIArgumentRange(MaterialInfo::wl, 620, 4700),
            MIComment("nr(wv) relation fitted by L. Piskorski (PLaSK developer), unpublished; "),
            MIComment("fitting data from 650-830nm and 1800-2560nm wavelength ranges; "),
            MIComment("basing on fig.5a from Adachi nR(wv) relation can be used for 620-4700nm wavelength range")
            )
double GaSb::nr(double wl, double T, double) const {
    double tE = phys::h_eVc1e9/wl; // wl -> E
    double nR300K = 0.502*tE*tE*tE - 1.216*tE*tE + 1.339*tE + 3.419;
    return ( nR300K + nR300K*8.2e-5*(T-300.) );
}

bool GaSb::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaSb> materialDB_register_GaSb;

}} // namespace plask::materials
