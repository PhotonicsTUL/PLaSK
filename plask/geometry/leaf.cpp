#include "leaf.h"
#include "manager.h"
#include "reader.h"

namespace plask {

//initialization common for all leafs
template <typename LeafType>
inline void setupLeaf(GeometryReader& reader, LeafType& leaf) {
    leaf.material = reader.manager.materialsDB.get(XML::requireAttr(reader.source, "material"));
    XML::requireTagEnd(reader.source, reader.source.getNodeName());
}

template <typename BlockType>
inline void setupBlock2d3d(GeometryReader& reader, BlockType& block) {
    block.size.tran = XML::requireAttr<double>(reader.source, reader.getAxisTranName());
    block.size.up = XML::requireAttr<double>(reader.source, reader.getAxisUpName());
    setupLeaf(reader, block);
}

GeometryElement* read_block2d(GeometryReader& reader) {
    std::unique_ptr< Block<2> > block(new Block<2>());
    setupBlock2d3d(reader, *block);
    return block.release();
}

GeometryElement* read_block3d(GeometryReader& reader) {
    std::unique_ptr< Block<3> > block(new Block<3>());
    block->size.lon = XML::requireAttr<double>(reader.source, reader.getAxisLonName());
    setupBlock2d3d(reader, *block);
    return block.release();
}

GeometryReader::RegisterElementReader block2d_reader("block2d", read_block2d);
GeometryReader::RegisterElementReader block3d_reader("block3d", read_block3d);

}   // namespace plask
