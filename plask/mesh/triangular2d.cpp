#include "triangular2d.h"

#include <boost/range/irange.hpp>

namespace plask {

Vec<3, double> TriangularMesh2D::Element::barycentric(Vec<2, double> p) const {
    // formula comes from https://codeplea.com/triangular-interpolation
    // but it is modified a bit, to reuse more diffs and call cross (which can be optimized)
    Vec<2, double>
            diff_2_3 = getNode(1) - getNode(2),
            diff_p_3 = p - getNode(2),
            diff_1_3 = getNode(0) - getNode(2);
    const double den = cross(diff_1_3, diff_2_3);   // diff_2_3.c1 * diff_1_3.c0 - diff_2_3.c0 * diff_1_3.c1
    Vec<3, double> res;
    res.c0 = cross(diff_p_3, diff_2_3) / den;   // (diff_2_3.c1 * diff_p_3.c0 - diff_2_3.c0 * diff_p_3.c1) / den
    res.c1 = cross(diff_1_3, diff_p_3) / den;   // (- diff_1_3.c1 * diff_p_3.c0 + diff_1_3.c0 * diff_p_3.c1) / den
    res.c2 = 1.0 - res.c0 - res.c1;
    return res;
}

Box2D TriangularMesh2D::Element::getBoundingBox() const {
    LocalCoords lo = getNode(0);
    LocalCoords up = lo;
    const LocalCoords B = getNode(1);
    if (B.c0 < lo.c0) lo.c0 = B.c0; else up.c0 = B.c0;
    if (B.c1 < lo.c1) lo.c1 = B.c1; else up.c1 = B.c1;
    const LocalCoords C = getNode(2);
    if (C.c0 < lo.c0) lo.c0 = C.c0; else if (C.c0 > up.c0) up.c0 = C.c0;
    if (C.c1 < lo.c1) lo.c1 = C.c1; else if (C.c1 > up.c1) up.c1 = C.c1;
    return Box2D(lo, up);
}

TriangularMesh2D::Builder::Builder(TriangularMesh2D &mesh): mesh(mesh) {
    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
        this->indexOfNode[mesh.nodes[i]] = i;
}

TriangularMesh2D::Builder::Builder(TriangularMesh2D &mesh, std::size_t predicted_number_of_elements, std::size_t predicted_number_of_nodes)
    : Builder(mesh)
{
    mesh.elementNodes.reserve(mesh.elementNodes.size() + predicted_number_of_elements);
    mesh.nodes.reserve(mesh.nodes.size() + predicted_number_of_nodes);
}

TriangularMesh2D::Builder::~Builder() {
    mesh.elementNodes.shrink_to_fit();
    mesh.nodes.shrink_to_fit();
}

TriangularMesh2D::Builder &TriangularMesh2D::Builder::add(TriangularMesh2D::LocalCoords p1, TriangularMesh2D::LocalCoords p2, TriangularMesh2D::LocalCoords p3) {
    mesh.elementNodes.push_back({addNode(p1), addNode(p2), addNode(p3)});
    return *this;
}

std::size_t TriangularMesh2D::Builder::addNode(TriangularMesh2D::LocalCoords node) {
    auto it = this->indexOfNode.emplace(node, mesh.nodes.size());
    if (it.second) // new element has been appended to the map
        this->mesh.nodes.push_back(node);
    return it.first->second;    // an index of the node (inserted or found)
}

struct ElementIndexValueGetter {
    TriangularMesh2D::ElementIndex::Rtree::value_type operator()(std::size_t index) const {
        TriangularMesh2D::Element el = this->src_mesh->getElement(index);
        const auto n0 = el.getNode(0);
        const auto n1 = el.getNode(1);
        const auto n2 = el.getNode(2);
        return std::make_pair(
                    TriangularMesh2D::ElementIndex::Box(
                        vec(std::min(std::min(n0.c0, n1.c0), n2.c0), std::min(std::min(n0.c1, n1.c1), n2.c1)),
                        vec(std::max(std::max(n0.c0, n1.c0), n2.c0), std::max(std::max(n0.c1, n1.c1), n2.c1))
                    ),
                    index);
    }

    const TriangularMesh2D* src_mesh;

    ElementIndexValueGetter(const TriangularMesh2D& src_mesh): src_mesh(&src_mesh) {}
};

TriangularMesh2D::ElementIndex::ElementIndex(const TriangularMesh2D &mesh)
    : mesh(mesh),
      rtree(makeFunctorIndexedIterator(ElementIndexValueGetter(mesh), 0),
            makeFunctorIndexedIterator(ElementIndexValueGetter(mesh), mesh.getElementsCount()))
{}

std::size_t TriangularMesh2D::ElementIndex::getIndex(Vec<2, double> p) const
{
    for (auto v: rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(p))) {
        const Element el = mesh.getElement(v.second);
        if (el.includes(p)) return v.second;
    }
    return INDEX_NOT_FOUND;
}

optional<TriangularMesh2D::Element> TriangularMesh2D::ElementIndex::getElement(Vec<2, double> p) const {
    for (auto v: rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(p))) {
        const Element el = mesh.getElement(v.second);
        if (el.includes(p)) return el;
    }
    return optional<TriangularMesh2D::Element>();
}

TriangularMesh2D TriangularMesh2D::masked(const TriangularMesh2D::Predicate &predicate) const {
    TriangularMesh2D result;
    Builder builder(result, elementNodes.size(), nodes.size());
    for (auto el: elements())
        if (predicate(el)) builder.add(el);
    return result;
}

void TriangularMesh2D::writeXML(XMLElement &object) const {
    object.attr("type", "triangular2d");
    // TODO write nodes and elements
}

static shared_ptr<Mesh> readTriangularMesh2D(XMLReader& reader) {
    shared_ptr<TriangularMesh2D> result = plask::make_shared<TriangularMesh2D>();
    // TODO read nodes and elements
    return result;
}

static RegisterMeshReader rectangular2d_reader("triangular2d", readTriangularMesh2D);



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
      elementIndex(*src_mesh)
{
}

template <typename DstT, typename SrcT>
DstT BarycentricTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    for (auto v: elementIndex.rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(wrapped_point))) {
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


// ------------------ Element mesh Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborElementTriangularMesh2DLazyDataImpl<DstT, SrcT>::NearestNeighborElementTriangularMesh2DLazyDataImpl(const shared_ptr<const TriangularMesh2D::ElementMesh> &src_mesh, const DataVector<const SrcT> &src_vec, const shared_ptr<const MeshD<2> > &dst_mesh, const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, TriangularMesh2D::ElementMesh, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      elementIndex(src_mesh->getOriginalMesh())
{
}

template<typename DstT, typename SrcT>
DstT NearestNeighborElementTriangularMesh2DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const {
    auto point = this->dst_mesh->at(index);
    auto wrapped_point = this->flags.wrap(point);
    std::size_t element_index = elementIndex.getIndex(point);
    if (element_index == TriangularMesh2D::ElementIndex::INDEX_NOT_FOUND)
        return NaN<decltype(this->src_vec[0])>();
    return this->flags.postprocess(point, this->src_vec[element_index]);
}



}   // namespace plask
