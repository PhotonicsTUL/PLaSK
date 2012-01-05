#include "leaf.h"
#include "manager.h"

namespace plask {

//initialization common for all leafs
template <typename LeafType>
inline void setupLeaf(GeometryManager& manager, XMLReader& source, LeafType& leaf) {
    leaf.material = manager.materialsDB.get(XML::requireAttr(source, "material"));
    XML::requireTagEnd(source);
}

template <typename BlockType>
inline void setupBlock2d3d(GeometryManager& manager, XMLReader& source, BlockType& block) {
//     block.size.x = XML::requireAttr<double>(source, "x");
//     block.size.y = XML::requireAttr<double>(source, "y");
    setupLeaf(manager, source, block);
}

GeometryElement* read_block2d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< Block<2> > block(new Block<2>());
    setupBlock2d3d(manager, source, *block);
    return block.release();
}

GeometryElement* read_block3d(GeometryManager& manager, XMLReader& source) {
    std::unique_ptr< Block<3> > block(new Block<3>());
//     block->size.z = XML::requireAttr<double>(source, "z");
    setupBlock2d3d(manager, source, *block);
    return block.release();
}

GeometryManager::RegisterElementReader block2d_reader("block2d", read_block2d);
GeometryManager::RegisterElementReader block3d_reader("block3d", read_block3d);

}   // namespace plask
