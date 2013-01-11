#include "align.h"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

namespace plask {
namespace align {

namespace details {

template <typename AlignerType, Primitive<3>::Direction dir>
inline void tryGetOneDirectionAligner(std::unique_ptr<OneDirectionAligner<dir>>& ans, boost::optional<double> param) {
    if (!param) return;
    if (ans) throw Exception("multiple specification of aligner in direction %1%", dir);
    ans.reset(new AlignerType(*param));
}

std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_TRAN>> transAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_TRAN>> result;
    tryGetOneDirectionAligner<Left>(result, dic(LEFT::value));
    tryGetOneDirectionAligner<Right>(result, dic(RIGHT::value));
    tryGetOneDirectionAligner<TranCenter>(result, dic(TRAN_CENTER::value));
    tryGetOneDirectionAligner<TranCenter>(result, dic("trancenter"));
    if (axis_name != "tran") tryGetOneDirectionAligner<TranCenter>(result, dic(axis_name + "center"));
    tryGetOneDirectionAligner<TranCenter>(result, dic(axis_name + "-center"));
    tryGetOneDirectionAligner<Tran>(result, dic(axis_name));
    return result;
}

std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_LONG>> lonAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_LONG>> result;
    tryGetOneDirectionAligner<Front>(result, dic(FRONT::value));
    tryGetOneDirectionAligner<Back>(result, dic(BACK::value));
    tryGetOneDirectionAligner<LonCenter>(result, dic(LON_CENTER::value));
    tryGetOneDirectionAligner<LonCenter>(result, dic("loncenter"));
    if (axis_name != "lon") tryGetOneDirectionAligner<LonCenter>(result, dic(axis_name + "center"));
    tryGetOneDirectionAligner<LonCenter>(result, dic(axis_name + "-center"));
    tryGetOneDirectionAligner<Lon>(result, dic(axis_name));
    return result;
}

std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_VERT>> vertAlignerFromDictionary(Dictionary dic, const std::string& axis_name) {
    std::unique_ptr<OneDirectionAligner<Primitive<3>::DIRECTION_VERT>> result;
    tryGetOneDirectionAligner<Top>(result, dic(TOP::value));
    tryGetOneDirectionAligner<Bottom>(result, dic(BOTTOM::value));
    tryGetOneDirectionAligner<VertCenter>(result, dic(VERT_CENTER::value));
    tryGetOneDirectionAligner<VertCenter>(result, dic("vertcenter"));
    if (axis_name != "vert") tryGetOneDirectionAligner<VertCenter>(result, dic(axis_name + "center"));
    tryGetOneDirectionAligner<VertCenter>(result, dic(axis_name + "-center"));
    tryGetOneDirectionAligner<Vert>(result, dic(axis_name));
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
