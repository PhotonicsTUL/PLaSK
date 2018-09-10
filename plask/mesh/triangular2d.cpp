#include "triangular2d.h"

namespace plask {

TriangularMesh2D::Builder::Builder(TriangularMesh2D &mesh): mesh(mesh) {
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        this->indexOfNode[mesh.nodes[i]] = i;
}

TriangularMesh2D::Builder &TriangularMesh2D::Builder::add(TriangularMesh2D::LocalCoords p1, TriangularMesh2D::LocalCoords p2, TriangularMesh2D::LocalCoords p3) {
    mesh.elementsNodes.push_back({addNode(p1), addNode(p2), addNode(p3)});
    return *this;
}

std::size_t TriangularMesh2D::Builder::addNode(TriangularMesh2D::LocalCoords node) {
    auto it = this->indexOfNode.emplace(node, mesh.nodes.size());
    if (it.second) // new element has been appended to the map
        this->mesh.nodes.push_back(node);
    return it.first->second;    // index of node (inserted or found)
}



}   // namespace plask
