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
    block.size.tran = reader.source.requireAttribute<double>(reader.getAxisTranName());
    block.size.up = reader.source.requireAttribute<double>(reader.getAxisUpName());
    setupLeaf(reader, block);
}

shared_ptr<GeometryElement> read_block2D(GeometryReader& reader) {
    shared_ptr< Block<2> > block(new Block<2>());
    setupBlock2D3D(reader, *block);
    return block;
}

shared_ptr<GeometryElement> read_block3D(GeometryReader& reader) {
    shared_ptr< Block<3> > block(new Block<3>());
    block->size.lon = reader.source.requireAttribute<double>(reader.getAxisLonName());
    setupBlock2D3D(reader, *block);
    return block;
}

static GeometryReader::RegisterElementReader block2D_reader("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_block2D);
static GeometryReader::RegisterElementReader rectangle_reader("rectangle", read_block2D);
static GeometryReader::RegisterElementReader block3D_reader("block" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_block3D);
static GeometryReader::RegisterElementReader cuboid_reader("cuboid", read_block3D);

shared_ptr<GeometryElement> changeToBlock(const shared_ptr<Material>& material, const shared_ptr<const GeometryElement>& to_change, Vec<3, double>& translation) {
    if (to_change->getDimensionsCount() == 3) {
        shared_ptr<const GeometryElementD<3>> el = static_pointer_cast<const GeometryElementD<3>>(to_change);
        Box3D bb = el->getBoundingBox();
        translation = bb.lower;
        return make_shared<Block<3>>(bb.size(), material);
    } else {    //to_change->getDimensionsCount() == 3
        shared_ptr<const GeometryElementD<2>> el = static_pointer_cast<const GeometryElementD<2>>(to_change);
        Box2D bb = el->getBoundingBox();
        translation = vec<3, double>(bb.lower);
        return make_shared<Block<2>>(bb.size(), material);
    }
}

}   // namespace plask
