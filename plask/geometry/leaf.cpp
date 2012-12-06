#include "leaf.h"
#include "reader.h"

namespace plask {

//initialization common for all leafs
template <typename LeafType>
inline void setupLeaf(GeometryReader& reader, LeafType& leaf) {
    leaf.material = reader.getMaterial(reader.source.requireAttribute("material"));
    //XML::requireTagEndOrEmptyTag(reader.source, reader.source.getNodeName());
    reader.source.requireTagEnd();
}

template <typename BlockType>
inline static void setupBlock2D3D(GeometryReader& reader, BlockType& block) {
    block.size.tran() = reader.source.requireAttribute<double>("d"+reader.getAxisTranName());
    block.size.vert() = reader.source.requireAttribute<double>("d"+reader.getAxisUpName());
    setupLeaf(reader, block);
}

template <>
void Block<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr(axes.getNameForTran(), size.tran())
                    .attr(axes.getNameForVert(), size.vert())
                    .attr("material", material->str());
}

template <>
void Block<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    dest_xml_object.attr(axes.getNameForLong(), size.lon())
                    .attr(axes.getNameForTran(), size.tran())
                    .attr(axes.getNameForVert(), size.vert())
                    .attr("material", material->str());
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
    block->size.lon() = reader.source.requireAttribute<double>("d"+reader.getAxisLonName());
    setupBlock2D3D(reader, *block);
    return block;
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
