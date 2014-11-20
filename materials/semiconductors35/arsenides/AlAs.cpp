#include "AlAs.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {


MI_PROPERTY(AlAs, lattC,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 5.6611 + 2.90e-5 * (T-300.);
    return tLattC;
}


MI_PROPERTY(AlAs, Eg,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.099, 0.885e-3, 530., T);
    else if (point == 'X') tEg = phys::Varshni(2.24, 0.70e-3, 530., T);
    else if (point == 'L') tEg = phys::Varshni(2.46, 0.605e-3, 204., T);
    else if (point == '*')
    {
        double tEgG = phys::Varshni(3.099, 0.885e-3, 530., T);
        double tEgX = phys::Varshni(2.24, 0.70e-3, 530., T);
        double tEgL = phys::Varshni(2.46, 0.605e-3, 204., T);
        tEg = min(tEgG,min(tEgX,tEgL));
    }
    if (!e) return tEg;
    else return ( CB(T,e,point) - max(VB(T,e,point,'H'),VB(T,e,point,'L')) );
}


MI_PROPERTY(AlAs, Dso,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::Dso(double T, double e) const {
    return 0.28;
}


MI_PROPERTY(AlAs, Me,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.230-232"),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlAs::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0., 0.);
    double tMeG(0.124), tMeX(0.71), tMeL(0.78);
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

MI_PROPERTY(AlAs, Mhh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlAs::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.51, 0.51); // [001]
    return tMhh;
}

MI_PROPERTY(AlAs, Mlh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence")
           )
Tensor2<double> AlAs::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.18, 0.18);
    return tMlh;
}

MI_PROPERTY(AlAs, Mh,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.235"),
            MIComment("no temperature dependence; "),
            MIComment("mh = (mhh^1.5+mlh^1.5)^(2/3)")
            )
Tensor2<double> AlAs::Mh(double T, double e) const {
    double tMc00 = pow(pow(Mhh(T,e).c00,1.5)+pow(Mlh(T,e).c00,1.5),(2./3.));
    double tMc11 = pow(pow(Mhh(T,e).c11,1.5)+pow(Mlh(T,e).c11,1.5),(2./3.));
    Tensor2<double> tMh(tMc00, tMc11); // [001]
    return ( tMh );
}

MI_PROPERTY(AlAs, CB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875")
           )
double AlAs::CB(double T, double e, char point) const {
    double tCB( VB(T,0.,point, 'H') + Eg(T,0.,point) );
    if (!e) return tCB;
    else return tCB + 2.*ac(T)*(1.-c12(T)/c11(T))*e;
}


MI_PROPERTY(AlAs, VB,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::VB(double T, double e, char point, char hole) const {
    double tVB(-1.33);
    if (e) {
        double DEhy = 2.*av(T)*(1.-c12(T)/c11(T))*e;
        double DEsh = -2.*b(T)*(1.+2.*c12(T)/c11(T))*e;
        if (hole=='H') return ( tVB + DEhy - 0.5*DEsh );
        else if (hole=='L') return ( tVB + DEhy -0.5*Dso(T,e) + 0.25*DEsh + 0.5*sqrt(Dso(T,e)*Dso(T,e)+Dso(T,e)*DEsh+2.25*DEsh*DEsh) );
        else return 0.;
    }
    return tVB;
}


MI_PROPERTY(AlAs, ac,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::ac(double T) const {
    return -5.64;
}


MI_PROPERTY(AlAs, av,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::av(double T) const {
    return 2.47;
}


MI_PROPERTY(AlAs, b,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::b(double T) const {
    return -2.3;
}


MI_PROPERTY(AlAs, d,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::d(double T) const {
    return -3.4;
}


MI_PROPERTY(AlAs, c11,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::c11(double T) const {
    return 125.0;
}


MI_PROPERTY(AlAs, c12,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::c12(double T) const {
    return 53.4;
}


MI_PROPERTY(AlAs, c44,
            MISource("I. Vurgaftman et al., J. Appl. Phys. 89 (2001) 5815-5875"),
            MIComment("no temperature dependence")
           )
double AlAs::c44(double T) const {
    return 54.2;
}


MI_PROPERTY(AlAs, thermk,
            MISource("S. Adachi, Properties of Semiconductor Alloys: Group-IV, III-V and II-VI Semiconductors, John Wiley and Sons (2009) p.67; "), // 300 K
            MISource("S. Adachi, Properties of Aluminium Gallium Arsenide, INSPEC (1993) p.48") // temperature dependence
           )
Tensor2<double> AlAs::thermk(double T, double t) const {
    double tk = 91. * pow((300./T),1.375);
    return(Tensor2<double>(tk, tk));
}

MI_PROPERTY(AlAs, dens,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.18"),
            MIComment("no temperature dependence")
            )
double AlAs::dens(double T) const { return 3.73016e3; }

MI_PROPERTY(AlAs, cp,
            MISource("S. Adachi, Properties of Semiconductors Alloys, John Wiley and Sons (2009) p.52"),
            MIComment("no temperature dependence")
            )
double AlAs::cp(double T) const { return 0.424e3; }


MI_PROPERTY(AlAs, nr,
            MISource("S. Gehrsitz, J. Appl. Phys. 87 (2000) 7825-7837; "),
            MISource("S. Adachi, Properties of Group-IV, III-V and II-VI Semiconductors, Wiley 2005"), // temperature dependence
            MIComment("fit by Lukasz Piskorski")
           )
double AlAs::nr(double wl, double T, double n) const {
    double L2 = wl*wl*1e-6;
    double nR296K = sqrt(1.+7.055*L2/(L2-0.068));
    return ( nR296K + nR296K*4.6e-5*(T-296.) );
}


MI_PROPERTY(AlAs, eps,
            MISource("http://www.iue.tuwien.ac.at/phd/quay/node27.html")
           )
double AlAs::eps(double T) const {
    return 10.1;
}

std::string AlAs::name() const { return NAME; }

bool AlAs::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<AlAs> materialDB_register_AlAs;

}}       // namespace plask::materials
