#include "axes.h"
#include "exceptions.h"
#include "utils/string.h"

namespace plask {

const AxisNames& AxisNames::Register::get(const std::string &name) const {
    auto i = axisNames.find(removedChars(name, ",._ \t"));
    if (i == axisNames.end())
        throw NoSuchAxisNames(name);
    return i->second;
}

AxisNames::Register AxisNames::axisNamesRegister =
        //c0, c1, c2, axis names:
        AxisNames::Register
        ("x", "y", "z", "yz", "z_up")
        ("z", "x", "y", "xy", "y_up")
        ("y", "z", "x", "zx", "x_up")
        ("p", "r", "z", "rz", "rad")
        ("l", "t", "v", "abs")
        ("long", "tran", "vert", "absolute");


AxisNames::AxisNames(const std::string& c0_name, const std::string& c1_name, const std::string& c2_name)
    : byIndex{c0_name, c1_name, c2_name} {}

std::size_t AxisNames::operator [](const std::string &name) const {
    if (name == byIndex[0] || name == "l" || name == "long") return 0;
    if (name == byIndex[1] || name == "t" || name == "tran") return 1;
    if (name == byIndex[2] || name == "v" || name == "vert") return 2;
    return 3;
}

Primitive<3>::Direction AxisNames::get3D(const std::string &name) const {
    std::size_t res = operator [] (name);
    if (res == 3) throw Exception("\"{0}\" is not proper axis name.", name);
    return Primitive<3>::Direction(res);
}

Primitive<2>::Direction AxisNames::get2D(const std::string &name) const {
    std::size_t res = operator [] (name);
    if (res == 0 || res == 3) throw Exception("\"{0}\" is not proper 2D axis name.", name);
    return Primitive<2>::Direction(res - 1);
}

std::string AxisNames::str() const {
    if (byIndex[0].length() == 1 && byIndex[1].length() == 1 && byIndex[2].length() == 1)
        return byIndex[0] + byIndex[1] + byIndex[2];
    return byIndex[0] + "," + byIndex[1] + "," + byIndex[2];
}

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
