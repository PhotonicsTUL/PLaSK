#include "../math.h"
#include "reader.h"
#include "background.h"

namespace plask {

template<>
shared_ptr<Material> Background<2>::getMaterial(const DVec& p) const {
    DVec r = p;
    Box bbox = getChild()->getBoundingBox();
    if (_extend & EXTEND_TRAN) {
        if (r.tran > bbox.upper.tran) r.tran = bbox.upper.tran;
        else if (r.tran < bbox.lower.tran) r.tran = bbox.lower.tran;
    }
    if (_extend & EXTEND_VERTICAL) {
        if (r.up > bbox.upper.up) r.up = bbox.upper.up;
        else if (r.up < bbox.lower.up) r.up = bbox.lower.up;
    }
    return getChild()->getMaterial(r);
}

template<>
shared_ptr<Material> Background<3>::getMaterial(const DVec& p) const {
    DVec r = p;
    Box bbox = getChild()->getBoundingBox();
    if (_extend & EXTEND_LON) {
        if (r.lon > bbox.upper.lon) r.lon = bbox.upper.lon;
        else if (r.lon < bbox.lower.lon) r.lon = bbox.lower.lon;
    }
    if (_extend & EXTEND_TRAN) {
        if (r.tran > bbox.upper.tran) r.tran = bbox.upper.tran;
        else if (r.tran < bbox.lower.tran) r.tran = bbox.lower.tran;
    }
    if (_extend & EXTEND_VERTICAL) {
        if (r.up > bbox.upper.up) r.up = bbox.upper.up;
        else if (r.up < bbox.lower.up) r.up = bbox.lower.up;
    }
    return getChild()->getMaterial(r);
}

/*
#define extend_attr "along"

static shared_ptr<GeometryElement> read_background2d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    int extend = Background<2>::EXTEND_NONE;
    std::string extend_str = XML::getAttribute<std::string>(reader.source, extend_attr, "all");
    if (extend_str == "all") {
        extend = Background<2>::EXTEND_ALL;
    } else {
        for (auto c: extend_str) {
            if (c == reader.getAxisTranName()[0]) extend |= Background<2>::EXTEND_TRAN;
            else if (c == reader.getAxisUpName()[0]) extend |= Background<2>::EXTEND_VERTICAL;
            else throw XMLBadAttrException(reader.source.getNodeName(), extend_attr, extend_str);
        }
    }
    shared_ptr<Background<2>> background(new Background<2>(Background<2>::ExtendType(extend)));
    background->setChild(reader.readExactlyOneChild<typename Background<2>::ChildType>());
    return background;
}

static shared_ptr<GeometryElement> read_background3d(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    int extend = Background<3>::EXTEND_NONE;
    std::string extend_str = XML::getAttribute<std::string>(reader.source, extend_attr, "all");
    if (extend_str == "all") {
        extend = Background<3>::EXTEND_ALL;
    } else {
        for (auto c: extend_str) {
            if (c == reader.getAxisLonName()[0]) extend |= Background<3>::EXTEND_LON;
            else if (c == reader.getAxisTranName()[0]) extend |= Background<3>::EXTEND_TRAN;
            else if (c == reader.getAxisUpName()[0]) extend |= Background<3>::EXTEND_VERTICAL;
            else throw XMLBadAttrException(reader.source.getNodeName(), extend_attr, extend_str);
        }
    }
    shared_ptr<Background<3>> background(new Background<3>(Background<3>::ExtendType(extend)));
    background->setChild(reader.readExactlyOneChild<typename Background<3>::ChildType>());
    return background;
}

static GeometryReader::RegisterElementReader background2d_reader("background" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_background2d);
static GeometryReader::RegisterElementReader background3d_reader("background" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_background3d);
*/


} // namespace plask
