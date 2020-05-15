#include "leaf.h"
#include "../manager.h"
#include "reader.h"

#define PLASK_BLOCK2D_NAME ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_BLOCK3D_NAME ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim>
XMLWriter::Element& GeometryObjectLeaf<dim>::SolidMaterial::writeXML(XMLWriter::Element& dest_xml_object,
                                                                     const AxisNames&) const {
    return material ? dest_xml_object.attr(GeometryReader::XML_MATERIAL_ATTR, material->str()) : dest_xml_object;
}

template <int dim>
XMLWriter::Element& GeometryObjectLeaf<dim>::GradientMaterial::writeXML(XMLWriter::Element& dest_xml_object,
                                                                        const AxisNames&) const {
    if (!materialFactory) return dest_xml_object;
    return dest_xml_object.attr(GeometryReader::XML_MATERIAL_BOTTOM_ATTR, (*materialFactory)(0.0)->str())
        .attr(GeometryReader::XML_MATERIAL_TOP_ATTR, (*materialFactory)(1.0)->str());
}

template <int dim> GeometryReader& GeometryObjectLeaf<dim>::readMaterial(GeometryReader& src) {
    auto top_attr = src.source.getAttribute(GeometryReader::XML_MATERIAL_TOP_ATTR);
    auto bottom_attr = src.source.getAttribute(GeometryReader::XML_MATERIAL_BOTTOM_ATTR);
    if (!top_attr && !bottom_attr) {
        if (src.source.hasAttribute(GeometryReader::XML_MATERIAL_GRADING_ATTR))
            src.source.throwException(format("'{}' attribute allowed only for layers with graded material",
                                             GeometryReader::XML_MATERIAL_GRADING_ATTR));
        if (src.materialsAreRequired) {
            this->setMaterialFast(src.getMaterial(src.source.requireAttribute(GeometryReader::XML_MATERIAL_ATTR)));
        } else if (plask::optional<std::string> matstr = src.source.getAttribute(GeometryReader::XML_MATERIAL_ATTR))
            this->setMaterialFast(src.getMaterial(*matstr));
    } else {
        if (!top_attr || !bottom_attr)
            src.source.throwException(format("If '{0}' or '{1}' attribute is given, the other one is also required",
                                             GeometryReader::XML_MATERIAL_TOP_ATTR,
                                             GeometryReader::XML_MATERIAL_BOTTOM_ATTR));
        double shape = src.source.getAttribute<double>(GeometryReader::XML_MATERIAL_GRADING_ATTR, 1.);
        this->setMaterialTopBottomCompositionFast(src.getMixedCompositionFactory(*top_attr, *bottom_attr, shape));
    }
    return src;
}

template <int dim> GeometryObject::Type GeometryObjectLeaf<dim>::getType() const { return GeometryObject::TYPE_LEAF; }

template <int dim>
shared_ptr<Material> GeometryObjectLeaf<dim>::getMaterial(const typename GeometryObjectLeaf<dim>::DVec& p) const {
    return this->contains(p) ? materialProvider->getMaterial(*this, p) : shared_ptr<Material>();
}

/*void GeometryObjectLeaf::getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>,
GeometryObjectLeaf::Box, GeometryObjectLeaf::DVec> > &dest, const PathHints *path) const { dest.push_back(
std::tuple<shared_ptr<const GeometryObject>, Box, DVec>(this->shared_from_this(), this->getBoundingBox(),
Primitive<dim>::ZERO_VEC) );
}*/

template <int dim>
void GeometryObjectLeaf<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                                    std::vector<typename GeometryObjectLeaf<dim>::Box>& dest,
                                                    const PathHints*) const {
    if (predicate(*this)) dest.push_back(this->getBoundingBox());
}

template <int dim>
void GeometryObjectLeaf<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate,
                                              std::vector<shared_ptr<const GeometryObject>>& dest,
                                              const PathHints* /*path*/) const {
    if (predicate(*this)) dest.push_back(this->shared_from_this());
}

template <int dim>
void GeometryObjectLeaf<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                                std::vector<typename GeometryObjectLeaf::DVec>& dest,
                                                const PathHints*) const {
    if (predicate(*this)) dest.push_back(Primitive<dim>::ZERO_VEC);
}

template <int dim> bool GeometryObjectLeaf<dim>::hasInSubtree(const GeometryObject& el) const { return &el == this; }

template <int dim>
GeometryObject::Subtree GeometryObjectLeaf<dim>::getPathsTo(const GeometryObject& el, const PathHints*) const {
    return GeometryObject::Subtree(&el == this ? this->shared_from_this() : shared_ptr<const GeometryObject>());
}

template <int dim>
GeometryObject::Subtree GeometryObjectLeaf<dim>::getPathsAt(const typename GeometryObjectLeaf<dim>::DVec& point,
                                                            bool) const {
    return GeometryObject::Subtree(this->contains(point) ? this->shared_from_this()
                                                         : shared_ptr<const GeometryObject>());
}

template <int dim> shared_ptr<GeometryObject> GeometryObjectLeaf<dim>::getChildNo(std::size_t) const {
    throw OutOfBoundsException("GeometryObjectLeaf::getChildNo", "child_no");
}

template <int dim>
shared_ptr<const GeometryObject> GeometryObjectLeaf<dim>::changedVersion(const GeometryObject::Changer& changer,
                                                                         Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    changer.apply(result, translation);
    return result;
}

template <int dim>
shared_ptr<GeometryObject> GeometryObjectLeaf<dim>::deepCopy(
    std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<GeometryObject> result = this->shallowCopy();
    copied[this] = result;
    return result;
}

template struct PLASK_API GeometryObjectLeaf<2>;
template struct PLASK_API GeometryObjectLeaf<3>;

// Read alternative attributes
inline static double readAlternativeAttrs(GeometryReader& reader, const std::string& attr1, const std::string& attr2) {
    auto value1 = reader.source.getAttribute<double>(attr1);
    auto value2 = reader.source.getAttribute<double>(attr2);
    if (value1) {
        if (value2) throw XMLConflictingAttributesException(reader.source, attr1, attr2);
        if (*value1 < 0.) throw XMLBadAttrException(reader.source, attr1, boost::lexical_cast<std::string>(*value1));
        return *value1;
    } else {
        if (!value2) {
            if (reader.manager.draft) return 0.0;
            else
                throw XMLNoAttrException(reader.source, format("{0}' or '{1}", attr1, attr2));
        }
        if (*value2 < 0.) throw XMLBadAttrException(reader.source, attr2, boost::lexical_cast<std::string>(*value2));
        return *value2;
    }
}

template <typename BlockType> inline static void setupBlock2D3D(GeometryReader& reader, BlockType& block) {
    block.size.tran() = readAlternativeAttrs(reader, "d" + reader.getAxisTranName(), "width");
    block.size.vert() = readAlternativeAttrs(reader, "d" + reader.getAxisVertName(), "height");
    block.readMaterial(reader);
    reader.source.requireTagEnd();
}

shared_ptr<GeometryObject> read_block2D(GeometryReader& reader) {
    shared_ptr<Block<2>> block(new Block<2>());
    setupBlock2D3D(reader, *block);
    return block;
}

shared_ptr<GeometryObject> read_block3D(GeometryReader& reader) {
    shared_ptr<Block<3>> block(new Block<3>());
    block->size.lon() = readAlternativeAttrs(reader, "d" + reader.getAxisLongName(), "length");
    setupBlock2D3D(reader, *block);
    return block;
}

template <> void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<2>::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
        .attr("d" + axes.getNameForTran(), size.tran())
        .attr("d" + axes.getNameForVert(), size.vert());
}

template <> void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    GeometryObjectLeaf<3>::writeXMLAttr(dest_xml_object, axes);
    materialProvider->writeXML(dest_xml_object, axes)
        .attr("d" + axes.getNameForLong(), size.lon())
        .attr("d" + axes.getNameForTran(), size.tran())
        .attr("d" + axes.getNameForVert(), size.vert());
}

template <int dim> const char* Block<dim>::NAME = dim == 2 ? PLASK_BLOCK2D_NAME : PLASK_BLOCK3D_NAME;

template <int dim> std::string Block<dim>::getTypeName() const { return NAME; }

template <int dim> typename Block<dim>::Box Block<dim>::getBoundingBox() const {
    return Block<dim>::Box(Primitive<dim>::ZERO_VEC, size);
}

template <int dim> bool Block<dim>::contains(const typename Block<dim>::DVec& p) const {
    return this->getBoundingBox().contains(p);
}

template <int dim>
void Block<dim>::addPointsAlong(std::set<double>& points,
                                Primitive<3>::Direction direction,
                                unsigned max_steps,
                                double min_step_size) const {
    assert(int(direction) >= 3 - dim && int(direction) <= 3);
    if (this->materialProvider->isUniform(direction)) {
        points.insert(0);
        points.insert(size[size_t(direction) - (3 - dim)]);
    } else {
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        double length = size[size_t(direction) - (3 - dim)];
        unsigned steps = min(unsigned(length / min_step_size), max_steps);
        double step = length / steps;
        for (unsigned i = 0; i <= steps; ++i) points.insert(i * step);
    }
}

template <>
void Block<2>::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const {
    typedef typename GeometryObjectD<2>::LineSegment Segment;
    std::vector<double> pts0, pts1;
    {
        std::set<double> ps;
        addPointsAlong(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_TRAN), max_steps, min_step_size);
        pts0.reserve(ps.size());
        pts0.insert(pts0.end(), ps.begin(), ps.end());
    }
    {
        std::set<double> ps;
        addPointsAlong(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_VERT), max_steps, min_step_size);
        pts1.reserve(ps.size());
        pts1.insert(pts1.end(), ps.begin(), ps.end());
    }
    for (size_t i1 = 0; i1 < pts1.size(); ++i1) {
        double p1 = pts1[i1];
        for (size_t i0 = 1; i0 < pts0.size(); ++i0)
            segments.insert(Segment(DVec(pts0[i0 - 1], p1), DVec(pts0[i0], p1)));
    }
    for (size_t i0 = 0; i0 < pts0.size(); ++i0) {
        double p0 = pts0[i0];
        for (size_t i1 = 1; i1 < pts1.size(); ++i1)
            segments.insert(Segment(DVec(p0, pts1[i1 - 1]), DVec(p0, pts1[i1])));
    }
}

template <>
void Block<3>::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                    unsigned max_steps,
                                    double min_step_size) const {
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    std::vector<double> pts0, pts1, pts2;
    {
        std::set<double> ps;
        addPointsAlong(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_LONG), max_steps, min_step_size);
        pts0.reserve(ps.size());
        pts0.insert(pts0.end(), ps.begin(), ps.end());
    }
    {
        std::set<double> ps;
        addPointsAlong(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_TRAN), max_steps, min_step_size);
        pts1.reserve(ps.size());
        pts1.insert(pts1.end(), ps.begin(), ps.end());
    }
    {
        std::set<double> ps;
        addPointsAlong(ps, Primitive<3>::Direction(Primitive<3>::DIRECTION_VERT), max_steps, min_step_size);
        pts2.reserve(ps.size());
        pts2.insert(pts2.end(), ps.begin(), ps.end());
    }
    for (size_t i2 = 0; i2 < pts2.size(); ++i2) {
        double p2 = pts1[i2];
        for (size_t i1 = 0; i1 < pts1.size(); ++i1) {
            double p1 = pts1[i1];
            for (size_t i0 = 1; i0 < pts0.size(); ++i0)
                segments.insert(Segment(DVec(pts0[i0 - 1], p1, p2), DVec(pts0[i0], p1, p2)));
        }
        for (size_t i0 = 0; i0 < pts0.size(); ++i0) {
            double p0 = pts0[i0];
            for (size_t i1 = 1; i1 < pts1.size(); ++i1)
                segments.insert(Segment(DVec(p0, pts1[i1 - 1], p2), DVec(p0, pts1[i1], p2)));
        }
    }
    for (size_t i1 = 0; i1 < pts1.size(); ++i1) {
        double p1 = pts1[i1];
        for (size_t i0 = 0; i0 < pts0.size(); ++i0) {
            double p0 = pts0[i0];
            for (size_t i2 = 1; i2 < pts2.size(); ++i2)
                segments.insert(Segment(DVec(p0, p1, pts2[i2 - 1]), DVec(p0, p1, pts2[i2])));
        }
    }
}

template struct PLASK_API Block<2>;
template struct PLASK_API Block<3>;

static GeometryReader::RegisterObjectReader block2D_reader(PLASK_BLOCK2D_NAME, read_block2D);
static GeometryReader::RegisterObjectReader rectangle_reader("rectangle", read_block2D);
static GeometryReader::RegisterObjectReader block3D_reader(PLASK_BLOCK3D_NAME, read_block3D);
static GeometryReader::RegisterObjectReader cuboid_reader("cuboid", read_block3D);

shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material,
                                         const shared_ptr<const GeometryObject>& to_change,
                                         Vec<3, double>& translation) {
    if (to_change->getDimensionsCount() == 3) {
        shared_ptr<const GeometryObjectD<3>> el = static_pointer_cast<const GeometryObjectD<3>>(to_change);
        Box3D bb = el->getBoundingBox();
        translation = bb.lower;
        return plask::make_shared<Block<3>>(bb.size(), material);
    } else {  // to_change->getDimensionsCount() == 3
        shared_ptr<const GeometryObjectD<2>> el = static_pointer_cast<const GeometryObjectD<2>>(to_change);
        Box2D bb = el->getBoundingBox();
        translation = vec<3, double>(bb.lower);
        return plask::make_shared<Block<2>>(bb.size(), material);
    }
}

}  // namespace plask
