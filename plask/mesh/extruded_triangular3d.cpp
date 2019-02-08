#include "extruded_triangular3d.h"

#include <boost/range/irange.hpp>

#include "../utils/interpolation.h"

namespace plask {

inline Vec<3, double> from_longTran_vert(const Vec<2, double>& longTran, const double& vert) {
    return vec(longTran.c0, longTran.c1, vert);
}

inline Vec<2, double> to_longTran(const Vec<3, double>& longTranVert) {
    return Vec<2, double>(longTranVert.c0, longTranVert.c1);
}

ExtrudedTriangularMesh3D::Element::Element(const ExtrudedTriangularMesh3D &mesh, std::size_t elementIndex)
    : mesh(mesh)
{
    if (mesh.vertFastest) {
        const std::size_t seg_size = mesh.vertAxis->size()-1;
        longTranIndex = elementIndex / seg_size;
        vertIndex = elementIndex % seg_size;
    } else {
        const std::size_t seg_size = mesh.longTranMesh.getElementsCount();
        longTranIndex = elementIndex % seg_size;
        vertIndex = elementIndex / seg_size;
    }
}

Vec<3, double> ExtrudedTriangularMesh3D::Element::getMidpoint() const {
    return from_longTran_vert(
        longTranElement().getMidpoint(),
        (mesh.vertAxis->at(vertIndex) + mesh.vertAxis->at(vertIndex+1)) * 0.5
    );
}

bool ExtrudedTriangularMesh3D::Element::includes(Vec<3, double> p) const {
    return mesh.vertAxis->at(vertIndex) <= p.vert() && p.vert() <= mesh.vertAxis->at(vertIndex+1)
            && longTranElement().includes(to_longTran(p));
}

Box3D ExtrudedTriangularMesh3D::Element::getBoundingBox() const {
    Box2D ltBox = longTranElement().getBoundingBox();
    return Box3D(
                from_longTran_vert(ltBox.lower, mesh.vertAxis->at(vertIndex)),
                from_longTran_vert(ltBox.upper, mesh.vertAxis->at(vertIndex+1))
           );
}

Vec<3, double> ExtrudedTriangularMesh3D::at(std::size_t index) const {
    if (vertFastest) {
        const std::size_t seg_size = vertAxis->size();
        return at(index / seg_size, index % seg_size);
    } else {
        const std::size_t seg_size = longTranMesh.size();
        return at(index % seg_size, index / seg_size);
    }
}

std::size_t ExtrudedTriangularMesh3D::size() const {
    return longTranMesh.size() * vertAxis->size();
}

bool ExtrudedTriangularMesh3D::empty() const {
    return longTranMesh.empty() || vertAxis->empty();
}

void ExtrudedTriangularMesh3D::writeXML(XMLElement &object) const {
    object.attr("type", "extruded_triangular2d");
    { auto a = object.addTag("vert"); vertAxis->writeXML(a); }
    { auto a = object.addTag("long_tran"); longTranMesh.writeXML(a); }
}

Vec<3, double> ExtrudedTriangularMesh3D::at(std::size_t longTranIndex, std::size_t vertIndex) const {
    return from_longTran_vert(longTranMesh[longTranIndex], vertAxis->at(vertIndex));
}


// ------------------ Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::NearestNeighborExtrudedTriangularMesh3DLazyDataImpl(
        const shared_ptr<const ExtrudedTriangularMesh3D> &src_mesh,
        const DataVector<const SrcT> &src_vec,
        const shared_ptr<const MeshD<3> > &dst_mesh,
        const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      nodesIndex(boost::irange(std::size_t(0), src_mesh->size()),
                 typename RtreeOfTriangularMesh2DNodes::parameters_type(),
                 TriangularMesh2DGetterForRtree(&src_mesh->longTranMesh))
{
}

template <typename DstT, typename SrcT>
DstT NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const
{
    const auto point = this->dst_mesh->at(index);
    const auto wrapped_point = this->flags.wrap(point);
    const auto wrapped_longTran = to_longTran(wrapped_point);
    for (auto v: nodesIndex | boost::geometry::index::adaptors::queried(boost::geometry::index::nearest(wrapped_longTran, 1)))
        return this->flags.postprocess(point,
                   this->src_vec[
                     this->src_mesh->index(
                        v, this->src_mesh->vertAxis->findNearestIndex(wrapped_point.vert())
                     )
                   ]
               );
    assert(false);
}

template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<double, double>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


// ------------------ Barycentric / Linear interpolation ---------------------

template<typename DstT, typename SrcT>
BarycentricExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::BarycentricExtrudedTriangularMesh3DLazyDataImpl(
        const shared_ptr<const ExtrudedTriangularMesh3D> &src_mesh,
        const DataVector<const SrcT> &src_vec,
        const shared_ptr<const MeshD<3> > &dst_mesh,
        const InterpolationFlags &flags
    )
    : InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      elementIndex(src_mesh->longTranMesh)
{
}

template <typename DstT, typename SrcT>
DstT BarycentricExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const {
    const auto point = this->dst_mesh->at(index);
    const auto wrapped_point = this->flags.wrap(point);
    const auto wrapped_longTran = to_longTran(wrapped_point);

    for (auto v: elementIndex.rtree | boost::geometry::index::adaptors::queried(boost::geometry::index::intersects(wrapped_longTran))) {
        const auto el = this->src_mesh->longTranMesh.getElement(v.second);
        const auto b = el.barycentric(wrapped_longTran);
        if (b.c0 < 0.0 || b.c1 < 0.0 || b.c2 < 0.0) continue; // wrapped_longTran is outside of the triangle

        const std::size_t lonTranIndex0 = el.getNodeIndex(0),
                          lonTranIndex1 = el.getNodeIndex(1),
                          lonTranIndex2 = el.getNodeIndex(2);

        size_t index_vert_lo, index_vert_hi;
        double vert_lo, vert_hi;
        bool invert_vert_lo, invert_vert_hi;
        prepareInterpolationForAxis(
                    *this->src_mesh->vertAxis, this->flags,
                    wrapped_point.vert(), 2 /*index of vert axis*/,
                    index_vert_lo, index_vert_hi,
                    vert_lo, vert_hi,
                    invert_vert_lo, invert_vert_hi);

        typename std::remove_const<typename std::remove_reference<decltype(this->src_vec[0])>::type>::type
            data_lo = b.c0 * this->src_vec[this->src_mesh->index(lonTranIndex0, index_vert_lo)] +
                      b.c1 * this->src_vec[this->src_mesh->index(lonTranIndex1, index_vert_lo)] +
                      b.c2 * this->src_vec[this->src_mesh->index(lonTranIndex2, index_vert_lo)],
            data_up = b.c0 * this->src_vec[this->src_mesh->index(lonTranIndex0, index_vert_hi)] +
                      b.c1 * this->src_vec[this->src_mesh->index(lonTranIndex1, index_vert_hi)] +
                      b.c2 * this->src_vec[this->src_mesh->index(lonTranIndex2, index_vert_hi)];

        if (invert_vert_lo) data_lo = this->flags.reflect(2, data_lo);
        if (invert_vert_hi) data_up = this->flags.reflect(2, data_up);

        return this->flags.postprocess(point, interpolation::linear(vert_lo, data_lo, vert_hi, data_up, wrapped_point.vert()));
    }
    return NaN<decltype(this->src_vec[0])>();
}

template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<double, double>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


// ------------------ Element mesh Nearest Neighbor interpolation ---------------------

template<typename DstT, typename SrcT>
NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl(
        const shared_ptr<const ExtrudedTriangularMesh3D::ElementMesh> &src_mesh,
        const DataVector<const SrcT> &src_vec,
        const shared_ptr<const MeshD<3> > &dst_mesh,
        const InterpolationFlags &flags)
    : InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D::ElementMesh, const SrcT>(src_mesh, src_vec, dst_mesh, flags),
      elementIndex(src_mesh->getOriginalMesh().longTranMesh)
{
}

template<typename DstT, typename SrcT>
DstT NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<DstT, SrcT>::at(std::size_t index) const {
    const auto point = this->dst_mesh->at(index);
    const auto wrapped_point = this->flags.wrap(point);

    if (wrapped_point.vert() < this->src_mesh->vertAxis[0] || this->src_mesh->vertAxis[this->src_mesh->vertAxis.size()] < wrapped_point.vert())
        return NaN<decltype(this->src_vec[0])>();

    const auto wrapped_longTran = to_longTran(wrapped_point);

    std::size_t longTran_element_index = elementIndex.getIndex(wrapped_point);
    if (longTran_element_index == TriangularMesh2D::ElementIndex::INDEX_NOT_FOUND)
        return NaN<decltype(this->src_vec[0])>();

    return this->flags.postprocess(point,
                                   this->src_vec[this->src_mesh.index(
                                        longTran_element_index,
                                        this->src_mesh->vertAxis.findUpIndex(wrapped_point.vert())-1
                                   )]);
}

template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<double, double>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<dcomplex, dcomplex>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Vec<2,double>, Vec<2,double>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Vec<2,dcomplex>, Vec<2,dcomplex>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Vec<3,double>, Vec<3,double>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Vec<3,dcomplex>, Vec<3,dcomplex>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Tensor2<double>, Tensor2<double>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Tensor2<dcomplex>, Tensor2<dcomplex>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Tensor3<double>, Tensor3<double>>;
template struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<Tensor3<dcomplex>, Tensor3<dcomplex>>;


}   // namespace plask
