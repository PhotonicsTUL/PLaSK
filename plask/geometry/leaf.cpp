#include "leaf.h"
#include "reader.h"
#include "../manager.h"

namespace plask {

template <int dim>
XMLWriter::Element& GeometryObjectLeaf<dim>::SolidMaterial::writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &) const {
    return material ? dest_xml_object.attr(GeometryReader::XML_MATERIAL_ATTR, material->str()) : dest_xml_object;
}

template <int dim>
XMLWriter::Element& GeometryObjectLeaf<dim>::MixedCompositionMaterial::writeXML(XMLWriter::Element &dest_xml_object, const AxisNames &) const {
    if (! materialFactory) return dest_xml_object;
    return dest_xml_object
            .attr(GeometryReader::XML_MATERIAL_BOTTOM_ATTR, (*materialFactory)(0.0)->str())
            .attr(GeometryReader::XML_MATERIAL_TOP_ATTR, (*materialFactory)(1.0)->str());
}

template <int dim>
GeometryReader &GeometryObjectLeaf<dim>::readMaterial(GeometryReader &src) {
    auto top_attr = src.source.getAttribute(GeometryReader::XML_MATERIAL_TOP_ATTR);
    auto bottom_attr = src.source.getAttribute(GeometryReader::XML_MATERIAL_BOTTOM_ATTR);
    if (!top_attr && !bottom_attr) {
        if (src.source.hasAttribute(GeometryReader::XML_MATERIAL_GRADING_ATTR))
            src.source.throwException(format("'{}' attribute allowed only for layers with graded material", GeometryReader::XML_MATERIAL_GRADING_ATTR));        if (src.materialsAreRequired) {
            this->setMaterialFast(src.getMaterial(src.source.requireAttribute(GeometryReader::XML_MATERIAL_ATTR)));
        } else if (boost::optional<std::string> matstr = src.source.getAttribute(GeometryReader::XML_MATERIAL_ATTR))
            this->setMaterialFast(src.getMaterial(*matstr));
    } else {
        if (!top_attr || !bottom_attr)
            src.source.throwException(format("If '{0}' or '{1}' attribute is given, the second one is also required",
                                                GeometryReader::XML_MATERIAL_TOP_ATTR, GeometryReader::XML_MATERIAL_BOTTOM_ATTR));
        double shape = src.source.getAttribute<double>(GeometryReader::XML_MATERIAL_GRADING_ATTR, 1.);
        this->setMaterialTopBottomCompositionFast(src.getMixedCompositionFactory(*top_attr, *bottom_attr, shape));
    }
    return src;
}

template <int dim>
bool GeometryObjectLeaf<dim>::isUniform(Primitive<3>::Direction direction) const {
    return materialProvider->isUniform(direction);
}

template <int dim>
GeometryObject::Type GeometryObjectLeaf<dim>::getType() const { return GeometryObject::TYPE_LEAF; }

template <int dim>
shared_ptr<Material> GeometryObjectLeaf<dim>::getMaterial(const typename GeometryObjectLeaf<dim>::DVec &p) const {
    return this->contains(p) ? materialProvider->getMaterial(*this, p) : shared_ptr<Material>();
}

/*void GeometryObjectLeaf::getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, GeometryObjectLeaf::Box, GeometryObjectLeaf::DVec> > &dest, const PathHints *path) const {
    dest.push_back( std::tuple<shared_ptr<const GeometryObject>, Box, DVec>(this->shared_from_this(), this->getBoundingBox(), Primitive<dim>::ZERO_VEC) );
}*/

template <int dim>
void GeometryObjectLeaf<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate &predicate, std::vector<typename GeometryObjectLeaf<dim>::Box> &dest, const PathHints *) const {
    if (predicate(*this))
        dest.push_back(this->getBoundingBox());
}

template <int dim>
void GeometryObjectLeaf<dim>::getObjectsToVec(const GeometryObject::Predicate &predicate, std::vector<shared_ptr<const GeometryObject> > &dest, const PathHints *path) const {
    if (predicate(*this)) dest.push_back(this->shared_from_this());
}

template <int dim>
void GeometryObjectLeaf<dim>::getPositionsToVec(const GeometryObject::Predicate &predicate, std::vector<typename GeometryObjectLeaf::DVec> &dest, const PathHints *) const {
    if (predicate(*this)) dest.push_back(Primitive<dim>::ZERO_VEC);
}

template <int dim>
bool GeometryObjectLeaf<dim>::hasInSubtree(const GeometryObject &el) const {
    return &el == this;
}

template <int dim>
GeometryObject::Subtree GeometryObjectLeaf<dim>::getPathsTo(const GeometryObject &el, const PathHints *) const {
    return GeometryObject::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
}

template <int dim>
GeometryObject::Subtree GeometryObjectLeaf<dim>::getPathsAt(const typename GeometryObjectLeaf<dim>::DVec &point, bool) const {
    return GeometryObject::Subtree( this->contains(point) ? this->shared_from_this() : shared_ptr<const GeometryObject>() );
}

template <int dim>
shared_ptr<GeometryObject> GeometryObjectLeaf<dim>::getChildNo(std::size_t) const {
    throw OutOfBoundsException("GeometryObjectLeaf::getChildNo", "child_no");
}

template <int dim>
shared_ptr<const GeometryObject> GeometryObjectLeaf<dim>::changedVersion(const GeometryObject::Changer &changer, Vec<3, double> *translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    changer.apply(result, translation);
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
            if (reader.manager.draft)
                return 0.0;
            else
                throw XMLNoAttrException(reader.source, format("{0}' or '{1}", attr1, attr2));
        }
        if (*value2 < 0.) throw XMLBadAttrException(reader.source, attr2, boost::lexical_cast<std::string>(*value2));
        return *value2;
    }
}

template <typename BlockType>
inline static void setupBlock2D3D(GeometryReader& reader, BlockType& block) {
    block.size.tran() = readAlternativeAttrs(reader, "d"+reader.getAxisTranName(), "width");
    block.size.vert() = readAlternativeAttrs(reader, "d"+reader.getAxisVertName(), "height");
    block.readMaterial(reader);
    reader.source.requireTagEnd();
}

shared_ptr<GeometryObject> read_block2D(GeometryReader& reader) {
    shared_ptr< Block<2> > block(new Block<2>());
    setupBlock2D3D(reader, *block);
    return block;
}

shared_ptr<GeometryObject> read_block3D(GeometryReader& reader) {
    shared_ptr< Block<3> > block(new Block<3>());
    block->size.lon() = readAlternativeAttrs(reader, "d"+reader.getAxisLongName(), "length");
    setupBlock2D3D(reader, *block);
    return block;
}

template <>
void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    materialProvider->writeXML(dest_xml_object, axes)
                    .attr("d"+axes.getNameForTran(), size.tran())
                    .attr("d"+axes.getNameForVert(), size.vert());
}

template <>
void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    materialProvider->writeXML(dest_xml_object, axes)
                    .attr("d"+axes.getNameForLong(), size.lon())
                    .attr("d"+axes.getNameForTran(), size.tran())
                    .attr("d"+axes.getNameForVert(), size.vert());
}

template <int dim>
const char* Block<dim>::NAME = dim == 2 ?
            ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D) :
            ("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);

template <int dim>
std::string Block<dim>::getTypeName() const { return NAME; }

template <int dim>
typename Block<dim>::Box Block<dim>::getBoundingBox() const {
    return Block<dim>::Box(Primitive<dim>::ZERO_VEC, size);
}

template <int dim>
bool Block<dim>::contains(const typename Block<dim>::DVec &p) const {
    return this->getBoundingBox().contains(p);
}

template struct PLASK_API Block<2>;
template struct PLASK_API Block<3>;

static GeometryReader::RegisterObjectReader block2D_reader(Block<2>::NAME, read_block2D);
static GeometryReader::RegisterObjectReader rectangle_reader("rectangle", read_block2D);
static GeometryReader::RegisterObjectReader block3D_reader(Block<3>::NAME, read_block3D);
static GeometryReader::RegisterObjectReader cuboid_reader("cuboid", read_block3D);

shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryObject>& to_change, Vec<3, double>& translation) {
    if (to_change->getDimensionsCount() == 3) {
        shared_ptr<const GeometryObjectD<3>> el = static_pointer_cast<const GeometryObjectD<3>>(to_change);
        Box3D bb = el->getBoundingBox();
        translation = bb.lower;
        return plask::make_shared<Block<3>>(bb.size(), material);
    } else {    //to_change->getDimensionsCount() == 3
        shared_ptr<const GeometryObjectD<2>> el = static_pointer_cast<const GeometryObjectD<2>>(to_change);
        Box2D bb = el->getBoundingBox();
        translation = vec<3, double>(bb.lower);
        return plask::make_shared<Block<2>>(bb.size(), material);
    }
}

}   // namespace plask
