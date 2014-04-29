#include "regular1d.h"

namespace plask {

void RegularAxis::writeXML(XMLElement &object) const {
    object.attr("type", "rectilinear1d").attr("start", first()).attr("stop", last()).attr("num", size());
}

bool RegularAxis::isIncreasing() const
{
    return step() >= 0;
}

}
