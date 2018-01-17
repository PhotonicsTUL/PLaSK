#include "air.h"
#include "../log/log.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Air::name() const { return NAME; }
Material::Kind Air::kind() const { return Material::NONE; }


double Air::A(double /*T*/) const { RETURN_MATERIAL_NAN(A) }

double Air::absp(double /*lam*/, double /*T*/) const { return 0.; }

double Air::B(double /*T*/) const { RETURN_MATERIAL_NAN(B) }

double Air::C(double /*T*/) const { RETURN_MATERIAL_NAN(C) }

double Air::CB(double /*T*/, double /*e*/, char /*point*/) const { RETURN_MATERIAL_NAN(CB) }

double Air::chi(double /*T*/, double /*e*/, char /*point*/) const { RETURN_MATERIAL_NAN(chi) }

MI_PROPERTY(Air, cond,
            MISource("S.D. Pawar et al., Journal of Geophysical Research, vol. 114, no. D2, id. D02205 (8 pp.), 2009"),
            MIComment("average value from (0.3-0.8)*10^-14 S/m")
           )
Tensor2<double> Air::cond(double /*T*/) const {
    double c = 0.55e-14;
    return Tensor2<double>(c, c);
}

Material::ConductivityType Air::condtype() const { return Material::CONDUCTIVITY_OTHER; }

double Air::D(double /*T*/) const { RETURN_MATERIAL_NAN(D) }

MI_PROPERTY(Air, dens,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 1")
           )
double Air::dens(double /*T*/) const { return 1.161; }

double Air::Dso(double /*T*/, double /*e*/) const { RETURN_MATERIAL_NAN(Dso) }

double Air::EactA(double /*T*/) const { RETURN_MATERIAL_NAN(EactA) }
double Air::EactD(double /*T*/) const { RETURN_MATERIAL_NAN(EactD) }

double Air::Eg(double /*T*/, double /*e*/, char /*point*/) const { RETURN_MATERIAL_NAN(Eg) }

double Air::eps(double /*T*/) const { return 1.; }

double Air::lattC(double /*T*/, char /*x*/) const { RETURN_MATERIAL_NAN(lattC) }

Tensor2<double> Air::Me(double /*T*/, double /*e*/, char /*point*/) const { RETURN_MATERIAL_NAN(Me) }

Tensor2<double> Air::Mh(double /*T*/, double /*e*/) const { RETURN_MATERIAL_NAN(Mh) }

Tensor2<double> Air::Mhh(double /*T*/, double /*e*/) const { RETURN_MATERIAL_NAN(Mhh) }

Tensor2<double> Air::Mlh(double /*T*/, double /*e*/) const { RETURN_MATERIAL_NAN(B) }

Tensor2<double> Air::mob(double /*T*/) const { RETURN_MATERIAL_NAN(mob) }

double Air::Mso(double /*T*/, double /*e*/) const { RETURN_MATERIAL_NAN(Mso) }

double Air::Nf(double /*T*/) const { RETURN_MATERIAL_NAN(Nf) }

double Air::Ni(double /*T*/) const { RETURN_MATERIAL_NAN(Ni) }

MI_PROPERTY(Air, nr,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 10, p. 224"),
            MIComment("using equation in source the calculated values are 1.0002-1.0003 for 200-2000nm wavelength range"),
            MIArgumentRange(MaterialInfo::T, 200, 2000)
           )
double Air::nr(double /*lam*/, double /*T*/, double /*n*/) const { return 1.; }

MI_PROPERTY(Air, cp,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 1")
           )
double Air::cp(double /*T*/) const { return 1.007e3; }

MI_PROPERTY(Air, thermk,
            MISource("D.R. Lide, ed., CRC Handbook of Chemistry and Physics, Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005, section 6, p. 175"),
            MIComment("fit by Lukasz Piskorski"),
            MIArgumentRange(MaterialInfo::T, 100, 600)
           )
Tensor2<double> Air::thermk(double T, double /*h*/) const {
    double tCondT = 0.0258*pow(T/300.,0.84);
    return Tensor2<double>(tCondT, tCondT);
}

double Air::VB(double /*T*/, double /*e*/, char /*point*/, char /*hole*/) const { RETURN_MATERIAL_NAN(VB) }

Tensor2<double> Air::mobe(double /*T*/) const { RETURN_MATERIAL_NAN(mobe) }

Tensor2<double> Air::mobh(double /*T*/) const { RETURN_MATERIAL_NAN(mobh) }

double Air::Na() const { RETURN_MATERIAL_NAN(Na) }

double Air::Nd() const { RETURN_MATERIAL_NAN(Nd) }

bool Air::isEqual(const Material &/*other*/) const {
    return true;
}

static MaterialsDB::Register<Air> materialDB_register_Air;

}} // namespace plask::materials
