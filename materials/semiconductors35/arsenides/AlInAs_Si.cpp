#include "AlInAs_Si.h"

#include <cmath>
#include <plask/material/db.h>  //MaterialsDB::Register
#include <plask/material/info.h>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlInAs_Si::name() const { return NAME; }

std::string AlInAs_Si::str() const { return StringBuilder("Al", Al)("In")("As").dopant("Si", ND); }

Material::ConductivityType AlInAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PARENT(AlInAs_Si, AlInAs)

AlInAs_Si::AlInAs_Si(const Material::Composition& Comp, DopingAmountType Type, double Val): AlInAs(Comp), mAlAs_Si(Type,Val), mInAs_Si(Type,Val)
{
}

bool AlInAs_Si::isEqual(const Material &other) const {
    const AlInAs_Si& o = static_cast<const AlInAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlInAs::isEqual(other);
}

static MaterialsDB::Register<AlInAs_Si> materialDB_register_AlInAs_Si;

}}       // namespace plask::materials
