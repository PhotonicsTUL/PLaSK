#include "align.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace plask {
namespace align {

namespace details {

template <Primitive<3>::Direction dir, Aligner1D<dir> AlignerType(double coordinate)>
inline void tryGetAligner1D(Aligner1D<dir>& ans, boost::optional<double> param) {
    if (!param) return;
    if (!ans.isNull()) throw Exception("multiple specification of aligner in direction %1%", dir);
    ans = AlignerType(*param);
}

Aligner1D<Primitive<3>::DIRECTION_TRAN> transAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    Aligner1D<Primitive<3>::DIRECTION_TRAN> result;
    tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, left>(result, dic(LEFT::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, right>(result, dic(RIGHT::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, tranCenter>(result, dic(TRAN_CENTER::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, tranCenter>(result, dic("trancenter"));
    if (axis_name != "tran") tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, tranCenter>(result, dic(axis_name + "center"));
//     tryGetAligner1D<TranCenter>(result, dic(axis_name + "-center"));
    tryGetAligner1D<Primitive<3>::DIRECTION_TRAN, tran>(result, dic(axis_name));
    return result;
}

Aligner1D<Primitive<3>::DIRECTION_LONG> lonAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    Aligner1D<Primitive<3>::DIRECTION_LONG> result;
    tryGetAligner1D<Primitive<3>::DIRECTION_LONG, front>(result, dic(FRONT::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_LONG, back>(result, dic(BACK::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_LONG, lonCenter>(result, dic(LON_CENTER::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_LONG, lonCenter>(result, dic("longcenter"));
    if (axis_name != "long") tryGetAligner1D<Primitive<3>::DIRECTION_LONG, lonCenter>(result, dic(axis_name + "center"));
//     tryGetAligner1D<LongCenter>(result, dic(axis_name + "-center"));
    tryGetAligner1D<Primitive<3>::DIRECTION_LONG, lon>(result, dic(axis_name));
    return result;
}

Aligner1D<Primitive<3>::DIRECTION_VERT> vertAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    Aligner1D<Primitive<3>::DIRECTION_VERT> result;
    tryGetAligner1D<Primitive<3>::DIRECTION_VERT, top>(result, dic(TOP::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_VERT, bottom>(result, dic(BOTTOM::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_VERT, vertCenter>(result, dic(VERT_CENTER::value));
    tryGetAligner1D<Primitive<3>::DIRECTION_VERT, vertCenter>(result, dic("vertcenter"));
    if (axis_name != "vert") tryGetAligner1D<Primitive<3>::DIRECTION_VERT, vertCenter>(result, dic(axis_name + "center"));
//     tryGetAligner1D<VertCenter>(result, dic(axis_name + "-center"));
    tryGetAligner1D<Primitive<3>::DIRECTION_VERT, vert>(result, dic(axis_name));
    return result;
}

}   // namespace details

/*Aligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>* aligner3DFromString(std::string str) {
    boost::algorithm::to_lower(str);
         if (str == "front left" || str == "fl" || str == "left front" || str == "lf") return new FrontLeft();
    else if (str == "center left" || str == "cl" || str == "left center" || str == "lc") return new CenterLeft();
    else if (str == "back left" || str == "bl" || str == "left back" || str == "lb") return new BackLeft();
    else if (str == "front center" || str == "fc" || str == "center front" || str == "lf") return new FrontCenter();
    else if (str == "center center" || str == "cc" || str == "center" || str == "c") return new CenterCenter();
    else if (str == "back center" || str == "bl" || str == "center back" || str == "lb") return new BackCenter();
    else if (str == "front right" || str == "fr" || str == "right front" || str == "rf") return new FrontRight();
    else if (str == "center right" || str == "cr" || str == "right center" || str == "rc") return new CenterRight();
    else if (str == "back right" || str == "br" || str == "right back" || str == "rb") return new BackRight();
    throw BadInput("alignerFromString", "Wrong aligner specification");
    return nullptr;
}*/



}   // namespace align
}   // namespace plask
