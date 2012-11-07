#include "axes.h"
#include "exceptions.h"
#include "utils/string.h"

namespace plask {

AxisNames::AxisNames(const std::string &c0_name, const std::string &c1_name, const std::string &c2_name)
    : byIndex{c0_name, c1_name, c2_name} {}

std::size_t AxisNames::operator [](const std::string &name) const {
    if (byIndex[0] == name) return 0;
    if (byIndex[1] == name) return 1;
    if (byIndex[2] == name) return 2;
    return 3;
}

std::string AxisNames::str() const {
    if (byIndex[0].length() == 1 && byIndex[1].length() == 1 && byIndex[2].length() == 1)
        return byIndex[0] + byIndex[1] + byIndex[2];
    return byIndex[0] + "," + byIndex[1] + "," + byIndex[2];
}

const AxisNames& AxisNames::Register::get(const std::string &name) const {
    auto i = axisNames.find(removedChars(name, ",._ \t"));
    if (i == axisNames.end())
        throw NoSuchAxisNames(name);
    return i->second;
}

AxisNames::Register AxisNames::axisNamesRegister =
        //c0, c1, c2, axis names:
        AxisNames::Register
        ("x", "y", "z", "yz", "se", "z_up")
        ("z", "x", "y", "xy", "ee", "y_up")
        ("p", "r", "z", "rz", "rad")
        ("l", "t", "v", "abs1")
        ("long", "tran", "vert", "absolute", "abs");

const AxisNames& AxisNames::getAbsoluteNames() {
    return axisNamesRegister.get("abs");
}

bool AxisNames::operator ==(const AxisNames &to_compare) const {
    return
        this->byIndex[0] == to_compare.byIndex[0] &&
        this->byIndex[1] == to_compare.byIndex[1] &&
        this->byIndex[2] == to_compare.byIndex[2];
}

} // namespace plask
