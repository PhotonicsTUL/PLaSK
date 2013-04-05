#include "leaf.h"
#include "reader.h"

namespace plask {

// Initialization common for all leafs
template <typename LeafType>
inline void setupLeaf(GeometryReader& reader, LeafType& leaf) {
    leaf.material = reader.requireMaterial();
    //XML::requireTagEndOrEmptyTag(reader.source, reader.source.getNodeName());
    reader.source.requireTagEnd();
}

// Read alternative attributes
inline static double readAlternativeAttrs(GeometryReader& reader, const std::string& attr1, const std::string& attr2) {
    auto value1 = reader.source.getAttribute<double>(attr1);
    auto value2 = reader.source.getAttribute<double>(attr2);
    if (value1) {
        if (value2) throw XMLConflictingAttributesException(reader.source, attr1, attr2);
        else return *value1;
    } else {
        if (!value2) throw XMLNoAttrException(reader.source, format("%1%' or '%2%", attr1, attr2));
        else return *value2;
    }
}

template <typename BlockType>
inline static void setupBlock2D3D(GeometryReader& reader, BlockType& block) {
    block.size.tran() = readAlternativeAttrs(reader, "d"+reader.getAxisTranName(), "width");
    block.size.vert() = readAlternativeAttrs(reader, "d"+reader.getAxisVertName(), "height");
    setupLeaf(reader, block);
}

template struct Block<2>;
template struct Block<3>;

shared_ptr<GeometryObject> read_block2D(GeometryReader& reader) {
    shared_ptr< Block<2> > block(new Block<2>());
    setupBlock2D3D(reader, *block);
    return block;
}

shared_ptr<GeometryObject> read_block3D(GeometryReader& reader) {
    shared_ptr< Block<3> > block(new Block<3>());
    block->size.lon() = readAlternativeAttrs(reader, "d"+reader.getAxisLongName(), "depth");
    setupBlock2D3D(reader, *block);
    return block;
}

template <>
void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr(axes.getNameForTran(), size.tran())
                    .attr(axes.getNameForVert(), size.vert())
                    .attr(GeometryReader::XML_MATERIAL_ATTR, material->str());
}

template <>
void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr(axes.getNameForLong(), size.lon())
                    .attr(axes.getNameForTran(), size.tran())
                    .attr(axes.getNameForVert(), size.vert())
                    .attr(GeometryReader::XML_MATERIAL_ATTR, material->str());
}

static GeometryReader::RegisterObjectReader block2D_reader(Block<2>::NAME, read_block2D);
static GeometryReader::RegisterObjectReader rectangle_reader("rectangle", read_block2D);
static GeometryReader::RegisterObjectReader block3D_reader(Block<3>::NAME, read_block3D);
static GeometryReader::RegisterObjectReader cuboid_reader("cuboid", read_block3D);

shared_ptr<GeometryObject> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryObject>& to_change, Vec<3, double>& translation) {
    if (to_change->getDimensionsCount() == 3) {
        shared_ptr<const GeometryObjectD<3>> el = static_pointer_cast<const GeometryObjectD<3>>(to_change);
        Box3D bb = el->getBoundingBox();
        translation = bb.lower;
        return make_shared<Block<3>>(bb.size(), material);
    } else {    //to_change->getDimensionsCount() == 3
        shared_ptr<const GeometryObjectD<2>> el = static_pointer_cast<const GeometryObjectD<2>>(to_change);
        Box2D bb = el->getBoundingBox();
        translation = vec<3, double>(bb.lower);
        return make_shared<Block<2>>(bb.size(), material);
    }
}

}   // namespace plask
