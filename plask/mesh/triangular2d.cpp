#include "triangular2d.h"

#include <boost/range/irange.hpp>

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



// ------------------ Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborTriangularMesh2DLazyDataImpl<DstT, SrcT>::NearestNeighborTriangularMesh2DLazyDataImpl(const shared_ptr<const TriangularMesh2D> &src_mesh, const DataVector<const SrcT> &src_vec, const shared_ptr<const MeshD<2> > &dst_mesh, const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      nodesIndex(boost::irange(std::size_t(0), src_mesh->size()),
                 typename Rtree::parameters_type(),
                 TriangularMesh2DGetter(src_mesh))
{
}

template <typename DstT, typename SrcT>
DstT NearestNeighborTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    for (auto v: nodesIndex | boost::geometry::index::adaptors::queried(boost::geometry::index::nearest(wrapped_point, 1)))
        return this->flags.postprocess(point, this->src_vec[v]);
    assert(false);
}

template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<double, double>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


// ------------------ Barycentric / Linear interpolation ---------------------

template<typename DstT, typename SrcT>
BarycentricTriangularMesh2DLazyDataImpl<DstT, SrcT>::BarycentricTriangularMesh2DLazyDataImpl(const shared_ptr<const TriangularMesh2D> &src_mesh, const DataVector<const SrcT> &src_vec, const shared_ptr<const MeshD<2> > &dst_mesh, const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      trianglesIndex(makeFunctorIndexedIterator(ValueGetter(src_mesh), 0), makeFunctorIndexedIterator(ValueGetter(src_mesh), src_mesh->getElementsCount()))
{
}

template <typename DstT, typename SrcT>
DstT BarycentricTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    for (auto v: trianglesIndex | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(wrapped_point))) {
        const auto el = this->src_mesh->getElement(v.second);
        const auto b = el.barycentric(wrapped_point);
        if (b.c0 < 0.0 || b.c1 < 0.0 || b.c2 < 0.0) continue; // wrapped_point is outside of the triangle
        return this->flags.postprocess(point,
                                       b.c0*this->src_vec[el.getNodeIndex(0)] +
                                       b.c1*this->src_vec[el.getNodeIndex(1)] +
                                       b.c2*this->src_vec[el.getNodeIndex(2)]);
    }
    return NaN<decltype(this->src_vec[0])>();
}

template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<double, double>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


}   // namespace plask
