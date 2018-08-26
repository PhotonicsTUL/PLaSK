#ifndef PLASK__TRIANGLE2D_H
#define PLASK__TRIANGLE2D_H

#include "mesh.h"
#include <array>

namespace plask {

struct TriangleMesh2D: public MeshD<2> {

    std::vector< Vec<2, double> > nodes;

    typedef std::array<std::size_t, 3> TriangleNodeIndexes;

    std::vector< TriangleNodeIndexes > elementsNodes;

    struct Element {
        TriangleNodeIndexes triangleNodes;
        TriangleMesh2D& mesh;   // for getting access to the nodes

        Vec<2, double> getNode(std::size_t index) const {
            return mesh.nodes[index];
        }
    };


};

}   // namespace plask

#endif // TRIANGLE2D_H
