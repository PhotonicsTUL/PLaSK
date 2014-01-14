#include "GaN.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaN::name() const { return NAME; }

MI_PROPERTY(GaN, cond,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MISource("G. Koblmuller et al., Appl. Phys. Lett. 91 (2007) 221905"),
            MIArgumentRange(MaterialInfo::T, 270, 400)
            )
Tensor2<double> GaN::cond(double T) const {
    double tCond = 255*pow((T/300.),-0.18);
    return (Tensor2<double>(tCond,tCond));
}

MI_PROPERTY(GaN, thermk,
            MISource("C. Mion et al., App. Phys. Lett. 89 (2006) 092123"),
            MIArgumentRange(MaterialInfo::T, 300, 450)
            )
Tensor2<double> GaN::thermk(double T, double t) const {
    double fun_t = pow((tanh(0.001529*pow(t,0.984))),0.12), //TODO change t to microns
           tCondT = 230*fun_t*pow((T/300.),-1.43);
    return(Tensor2<double>(tCondT,tCondT));
 }

MI_PROPERTY(GaN, absp,
            MISource("J. Piprek et al., Proc. SPIE 6766 (2007) 67660H"),
            MIComment("fit to GaN:Si/GaN:Mg/GaN:undoped in region 360 - 400 nm"),
            MIComment("no temperature dependence")
            )
double GaN::absp(double wl, double T) const {
    double a = phys::h_eVc1e9/wl - Eg(T, 0., 'G');
    return ( 19000*exp(a/0.019) + 330*exp(a/0.07) );
}

MI_PROPERTY(GaN, nr,
            MISource("fit - Maciej Kuc"),
            MIArgumentRange(MaterialInfo::wl, 300, 580),
            MIComment("no temperature dependence")
            )
double GaN::nr(double wl, double T, double n) const {
    double dEg = Eg(T,0.,'G') - Eg(300.,0.,'G'),
           Eold = phys::h_eVc1e9 / wl,
           Enew = Eold - dEg;

    if (Enew > 1.000 && Enew < 2.138) // 580-1240 nm
        return ( 0.013914*Enew*Enew*Enew*Enew - 0.096422*Enew*Enew*Enew + 0.27318*Enew*Enew - 0.27725*Enew + 2.3535 );
    else if (Enew < 3.163) // 392-580 nm
        return ( 0.1152*Enew*Enew*Enew - 0.7955*Enew*Enew + 1.959*Enew + 0.68 );
    else if (Enew < 3.351) // 370-392 nm
        return ( 18.2292*Enew*Enew*Enew - 174.6974*Enew*Enew + 558.535*Enew - 593.164 );
    else if (Enew < 3.532) // 351-370 nm
        return ( 33.63905*Enew*Enew*Enew - 353.1446*Enew*Enew + 1235.0168*Enew - 1436.09 );
    else if (Enew < 4.100) // 336-351 nm
        return ( -0.72116*Enew*Enew*Enew + 8.8092*Enew*Enew - 35.8878*Enew + 51.335 );
    else if (Enew < 5.000) // 248-336 nm
        return ( 0.351664*Enew*Enew*Enew*Enew - 6.06337*Enew*Enew*Enew + 39.2317*Enew*Enew - 112.865*Enew + 124.358 );
    else
        return 0.;
    /*if (dopant == "Si" || dopant == "O") nR = nR*(1 - 1.05e-28*free_carrier_concentration(T,N,dopant));
    for GaN only!!!!*/
}

MI_PROPERTY(GaN, lattC,
            MISource("S. Adachi et al., Properties of Semiconductor Alloys: Group-IV, III–V and II–VI Semiconductors, Wiley 2009")
            )
double GaN::lattC(double T, char x) const {
    double tLattC(0.);
    if (x == 'a') tLattC = 3.1896;
    else if (x == 'c') tLattC = 5.1855;
    return (tLattC);
}

MI_PROPERTY(GaN, Eg,
            MISource("Vurgaftman et al. in Piprek 2007 Nitride Semicondcuctor Devices")
            )
double GaN::Eg(double T, double e, char point) const {
    double tEg(0.);
    if (point == 'G') tEg = phys::Varshni(3.510,0.914e-3,825.,T);
    return (tEg);
}

MI_PROPERTY(GaN, Me,
            MISource("Adachi WILEY 2009"),
            MIComment("no temperature dependence")
            )
Tensor2<double> GaN::Me(double T, double e, char point) const {
    Tensor2<double> tMe(0.,0.);
    if (point == 'G') {
        tMe.c00 = 0.22;
        tMe.c11 = 0.21;
    }
    return (tMe);
}

MI_PROPERTY(GaN, Mhh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mhh(double T, double e) const {
    Tensor2<double> tMhh(0.,0.);
    tMhh.c00 = 1.67;
    tMhh.c11 = 1.64;
    return (tMhh);
}

MI_PROPERTY(GaN, Mlh,
            MISeeClass<GaN>(MaterialInfo::Me)
            )
Tensor2<double> GaN::Mlh(double T, double e) const {
    Tensor2<double> tMlh(0.,0.);
    tMlh.c00 = 1.67;
    tMlh.c11 = 0.15;
    return (tMlh);
}

bool GaN::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<GaN> materialDB_register_GaN;

}}       // namespace plask::materials
