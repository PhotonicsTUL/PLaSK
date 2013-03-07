#include "air.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask {

std::string Air::name() const { return NAME; }
Material::Kind Air::kind() const { return Material::NONE; }


double Air::A(double T) const { throwNotApplicable("A(double T)"); return 0; }

double Air::absp(double wl, double T) const { return 0.; }

double Air::B(double T) const { throwNotApplicable("B(double T)"); return 0; }

double Air::C(double T) const { throwNotApplicable("C(double T)"); return 0; }

double Air::CBO(double T, char point) const { throwNotApplicable("CBO(double T, char point)"); return 0; }

double Air::chi(double T, char point) const { throwNotApplicable("chi(double T, char point)"); return 0; }

Tensor2<double> Air::cond(double T) const { return 1e-12; }

Material::ConductivityType Air::condtype() const { return Material::CONDUCTIVITY_OTHER; }

double Air::D(double T) const { throwNotApplicable("D(double T)"); return 0; }

double Air::dens(double T) const { throwNotApplicable("dens(double T)"); return 0; }

double Air::Dso(double T) const { throwNotApplicable("Dso(double T)"); return 0; }

double Air::EactA(double T) const { throwNotApplicable("EactA(double T)"); return 0; }
double Air::EactD(double T) const { throwNotApplicable("EactD(double T)"); return 0; }

double Air::Eg(double T, char point) const { throwNotApplicable("Eg(double T, char point)"); return 0; }

double Air::eps(double T) const { return 1.0; }

double Air::lattC(double T, char x) const { throwNotApplicable("lattC(double T, char x)"); return 0; }

Tensor2<double> Air::Me(double T, char point) const { throwNotApplicable("Me(double T, char point)"); return 0.; }

Tensor2<double> Air::Mh(double T, char EqType) const { throwNotApplicable("Mh(double T, char EqType)"); return 0.; }

Tensor2<double> Air::Mhh(double T, char point) const { throwNotApplicable("Mhh(double T, char point)"); return 0.; }

Tensor2<double> Air::Mlh(double T, char point) const { throwNotApplicable("B(double T)"); return 0.; }

Tensor2<double> Air::mob(double T) const { throwNotApplicable("mob(double T)"); return 0.; }

double Air::Mso(double T) const { throwNotApplicable("Mso(double T)"); return 0; }

double Air::Nc(double T, char point) const { throwNotApplicable("Nc(double T, char point)"); return 0; }
double Air::Nc(double T) const { throwNotApplicable("Nc(double T)"); return 0; }

double Air::Nf(double T) const { throwNotApplicable("Nf(double T)"); return 0; }

double Air::Ni(double T) const { throwNotApplicable("Ni(double T)"); return 0; }

double Air::nr(double wl, double T) const { return 1.; }

double Air::cp(double T) const { throwNotApplicable("cp(double T)"); return 0; }

Tensor2<double> Air::thermk(double T, double h) const { return 5e-5; }

double Air::VBO(double T) const { throwNotApplicable("VBO(double T)"); return 0; }

bool Air::isEqual(const Material &other) const {
    return true;
}

static MaterialsDB::Register<Air> materialDB_register_Air;

} // namespace plask
