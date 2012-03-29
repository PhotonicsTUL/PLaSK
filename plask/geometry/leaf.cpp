#include "leaf.h"
#include "manager.h"
#include "reader.h"

namespace plask {

//initialization common for all leafs
template <typename LeafType>
inline void setupLeaf(GeometryReader& reader, LeafType& leaf) {
    leaf.material = reader.materialsDB.get(XML::requireAttr(reader.source, "material"));
    XML::requireTagEndOrEmptyTag(reader.source, reader.source.getNodeName());
}

template <typename BlockType>
inline void setupBlock2d3d(GeometryReader& reader, BlockType& block) {
    block.size.tran = XML::requireAttr<double>(reader.source, reader.getAxisTranName());
    block.size.up = XML::requireAttr<double>(reader.source, reader.getAxisUpName());
    setupLeaf(reader, block);
}

shared_ptr<GeometryElement> read_block2d(GeometryReader& reader) {
    shared_ptr< Block<2> > block(new Block<2>());
    setupBlock2d3d(reader, *block);
    return block;
}

shared_ptr<GeometryElement> read_block3d(GeometryReader& reader) {
    shared_ptr< Block<3> > block(new Block<3>());
    block->size.lon = XML::requireAttr<double>(reader.source, reader.getAxisLonName());
    setupBlock2d3d(reader, *block);
    return block;
}

static GeometryReader::RegisterElementReader block2d_reader("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_block2d);
static GeometryReader::RegisterElementReader rectangle_reader("rectangle", read_block2d);
static GeometryReader::RegisterElementReader block3d_reader("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_block3d);
static GeometryReader::RegisterElementReader cuboid_reader("cuboid", read_block3d);

shared_ptr<GeometryElement> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryElement>& to_change, Vec<3, double>& translation) {
    if (to_change->getDimensionsCount() == 3) {
        shared_ptr<const GeometryElementD<3>> el = static_pointer_cast<const GeometryElementD<3>>(to_change);
        Box3d bb = el->getBoundingBox();
        translation = bb.lower;
        return make_shared<Block<3>>(bb.size(), material);
    } else {    //to_change->getDimensionsCount() == 3
        shared_ptr<const GeometryElementD<2>> el = static_pointer_cast<const GeometryElementD<2>>(to_change);
        Box2d bb = el->getBoundingBox();
        translation = vec<3, double>(bb.lower);
        return make_shared<Block<2>>(bb.size(), material);
    }
}

}   // namespace plask
