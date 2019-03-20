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

inline Box2D to_longTran(const Box3D& box) {
    return Box2D(to_longTran(box.lower), to_longTran(box.upper));
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

std::pair<std::size_t, std::size_t> ExtrudedTriangularMesh3D::longTranAndVertIndices(std::size_t index) const {
    if (vertFastest) {
        const std::size_t seg_size = vertAxis->size();
        return std::pair<std::size_t, std::size_t>(index / seg_size, index % seg_size);
    } else {
        const std::size_t seg_size = longTranMesh.size();
        return std::pair<std::size_t, std::size_t>(index % seg_size, index / seg_size);
    }
}

std::size_t ExtrudedTriangularMesh3D::vertIndex(std::size_t index) const {
    return vertFastest ? index % vertAxis->size() : index / longTranMesh.size();
}

TriangularMesh2D::SegmentsCounts ExtrudedTriangularMesh3D::countSegmentsIn(
        std::size_t layer,
        const GeometryD<3> &geometry,
        const GeometryObject &object,
        const PathHints *path) const
{
    TriangularMesh2D::SegmentsCounts result;
    for (const auto el: this->longTranMesh.elements())
        if (geometry.objectIncludes(object, path, this->at(el.getNodeIndex(0), layer)) &&
                geometry.objectIncludes(object, path, this->at(el.getNodeIndex(1), layer)) &&
                geometry.objectIncludes(object, path, this->at(el.getNodeIndex(2), layer)))
            this->longTranMesh.countSegmentsOf(result, el);
    return result;
}

template<ExtrudedTriangularMesh3D::SideBoundaryDir boundaryDir>
std::set<std::size_t> ExtrudedTriangularMesh3D::boundaryNodes(
        const ExtrudedTriangularMesh3D::LayersIntervalSet& layers,
        const GeometryD<3>& geometry,
        const GeometryObject& object,
        const PathHints *path) const
{
    std::set<std::size_t> result;
    for (ExtrudedTriangularMesh3D::LayersInterval layer_interval: layers) {
        for (std::size_t layer = layer_interval.lower(); layer < layer_interval.upper(); ++layer) {
            for (std::size_t longTranNode: this->longTranMesh.boundaryNodes<ExtrudedTriangularMesh3D::boundaryDir3Dto2D(boundaryDir)>(countSegmentsIn(layer, geometry, object, path)))
                result.insert(index(longTranNode, layer));
        }
    }
    return result;
}

ExtrudedTriangularMesh3D::LayersInterval ExtrudedTriangularMesh3D::layersIn(const Box3D &box) const {
    return LayersInterval(this->vertAxis->findIndex(box.lower.vert()), this->vertAxis->findUpIndex(box.upper.vert()));
}

ExtrudedTriangularMesh3D::LayersIntervalSet ExtrudedTriangularMesh3D::layersIn(const std::vector<Box3D>& boxes) const {
    LayersIntervalSet layers;
    for (const Box3D& box: boxes) {
        LayersInterval interval = layersIn(box);
        if (interval.lower() < interval.upper()) // if interval is not empty
            layers.add(interval);
    }
    return layers;
}

template <ExtrudedTriangularMesh3D::SideBoundaryDir boundaryDir>
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getObjBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        LayersIntervalSet layers = mesh.layersIn(geometry->getObjectBoundingBoxes(object, path));
        if (layers.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<boundaryDir>(layers, *geometry, *object, &path)));
    } );
}

template <ExtrudedTriangularMesh3D::SideBoundaryDir boundaryDir>
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getObjBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        LayersIntervalSet layers = mesh.layersIn(geometry->getObjectBoundingBoxes(object));
        if (layers.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new StdSetBoundaryImpl(mesh.boundaryNodes<boundaryDir>(layers, *geometry, *object)));
    } );
}

template<ExtrudedTriangularMesh3D::SideBoundaryDir boundaryDir>
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getMeshBoundary()
{
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new ExtrudedTriangularBoundaryImpl(
                 mesh,
                 mesh.longTranMesh.boundaryNodes<ExtrudedTriangularMesh3D::boundaryDir3Dto2D(boundaryDir)>(mesh.longTranMesh.countSegments()),
                 LayersInterval(0, mesh.vertAxis->size()-1)));
    } );
}

template<ExtrudedTriangularMesh3D::SideBoundaryDir boundaryDir>
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBoxBoundary(const Box3D &box)
{
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        LayersInterval layers = mesh.layersIn(box);
        if (layers.lower() >= layers.upper()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new ExtrudedTriangularBoundaryImpl(
                 mesh,
                 mesh.longTranMesh.boundaryNodes<ExtrudedTriangularMesh3D::boundaryDir3Dto2D(boundaryDir)>(mesh.longTranMesh.countSegmentsIn(to_longTran(box))),
                 layers));
    } );
}

BoundaryNodeSet ExtrudedTriangularMesh3D::topOrBottomBoundaryNodeSet(const Box3D &box, bool top) const {
    LayersInterval layers = layersIn(box);
    if (layers.lower() >= layers.upper()) return new EmptyBoundaryImpl();
    const std::size_t layer = top ? layers.upper()-1 : layers.lower();
    std::set<std::size_t> nodes3d;
    Box2D box2d = to_longTran(box);
    for (std::size_t index2d = 0; index2d < longTranMesh.size(); ++index2d)
        if (box2d.contains(longTranMesh[index2d]))
            nodes3d.insert(index(index2d, layer));
    return new StdSetBoundaryImpl(std::move(nodes3d));
}

BoundaryNodeSet ExtrudedTriangularMesh3D::topOrBottomBoundaryNodeSet(const GeometryD<3>& geometry, const GeometryObject& object, const PathHints *path, bool top) const {
    if (this->empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
    LayersIntervalSet layers = layersIn(geometry.getObjectBoundingBoxes(object, path));
    if (layers.empty()) return new EmptyBoundaryImpl();
    std::unique_ptr<std::size_t[]> index2d_to_layer(new std::size_t[this->longTranMesh.size()]);
    std::fill_n(index2d_to_layer.get(), this->longTranMesh.size(), std::numeric_limits<std::size_t>::max());
    if (top) {
        for (ExtrudedTriangularMesh3D::LayersInterval layer_interval: layers)
            for (std::size_t layer = layer_interval.lower(); layer < layer_interval.upper(); ++layer)
                for (std::size_t node2d_index = 0; node2d_index < longTranMesh.size(); ++node2d_index)
                    if (geometry.objectIncludes(object, path, at(node2d_index, layer)))
                        index2d_to_layer[node2d_index] = layer;
    } else {
        for (auto layers_it = layers.rbegin(); layers_it != layers.rend(); ++layers_it)
            for (std::size_t layer = layers_it->upper(); layer-- > layers_it->lower(); )
                for (std::size_t node2d_index = 0; node2d_index < longTranMesh.size(); ++node2d_index)
                    if (geometry.objectIncludes(object, path, at(node2d_index, layer)))
                        index2d_to_layer[node2d_index] = layer;
    }
    std::set<std::size_t> nodes3d;
    for (std::size_t node2d_index = 0; node2d_index < longTranMesh.size(); ++node2d_index)
        if (index2d_to_layer[node2d_index] != std::numeric_limits<std::size_t>::max())
            nodes3d.insert(index(node2d_index, index2d_to_layer[node2d_index]));
    return new StdSetBoundaryImpl(std::move(nodes3d));
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBackBoundary() {
    return getMeshBoundary<SideBoundaryDir::BACK>();
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getFrontBoundary() {
    return getMeshBoundary<SideBoundaryDir::FRONT>();
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getLeftBoundary() {
    return getMeshBoundary<SideBoundaryDir::LEFT>();
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getRightBoundary() {
    return getMeshBoundary<SideBoundaryDir::RIGHT>();
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBottomBoundary() {
    return Boundary( [](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new ExtrudedTriangularWholeLayerBoundaryImpl(mesh, 0));
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getTopBoundary() {
    return Boundary( [](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        if (mesh.empty()) return BoundaryNodeSet(new EmptyBoundaryImpl());
        return BoundaryNodeSet(new ExtrudedTriangularWholeLayerBoundaryImpl(mesh, mesh.vertAxis->size()-1));
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getAllSidesBoundary() {
    return getMeshBoundary<SideBoundaryDir::ALL>();
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBackOfBoundary(const Box3D &box) {
    return getBoxBoundary<SideBoundaryDir::BACK>(box);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getFrontOfBoundary(const Box3D &box) {
    return getBoxBoundary<SideBoundaryDir::FRONT>(box);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getLeftOfBoundary(const Box3D &box) {
    return getBoxBoundary<SideBoundaryDir::LEFT>(box);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getRightOfBoundary(const Box3D &box) {
    return getBoxBoundary<SideBoundaryDir::RIGHT>(box);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBottomOfBoundary(const Box3D &box) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        return mesh.topOrBottomBoundaryNodeSet(box, false);
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getTopOfBoundary(const Box3D &box) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>&) {
        return mesh.topOrBottomBoundaryNodeSet(box, true);
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getAllSidesOfBoundary(const Box3D &box) {
    return getBoxBoundary<SideBoundaryDir::ALL>(box);
}


ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBackOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return getObjBoundary<SideBoundaryDir::BACK>(object, path);
}
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBackOfBoundary(shared_ptr<const GeometryObject> object) {
    return getObjBoundary<SideBoundaryDir::BACK>(object);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getFrontOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return getObjBoundary<SideBoundaryDir::FRONT>(object, path);
}
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getFrontOfBoundary(shared_ptr<const GeometryObject> object) {
    return getObjBoundary<SideBoundaryDir::FRONT>(object);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return getObjBoundary<SideBoundaryDir::LEFT>(object, path);
}
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getLeftOfBoundary(shared_ptr<const GeometryObject> object) {
    return getObjBoundary<SideBoundaryDir::LEFT>(object);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return getObjBoundary<SideBoundaryDir::RIGHT>(object, path);
}
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getRightOfBoundary(shared_ptr<const GeometryObject> object) {
    return getObjBoundary<SideBoundaryDir::RIGHT>(object);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getAllSidesBoundaryIn(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return getObjBoundary<SideBoundaryDir::ALL>(object, path);
}
ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getAllSidesBoundaryIn(shared_ptr<const GeometryObject> object) {
    return getObjBoundary<SideBoundaryDir::ALL>(object);
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getTopOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        return mesh.topOrBottomBoundaryNodeSet(*geometry, *object, &path, true);
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getTopOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        return mesh.topOrBottomBoundaryNodeSet(*geometry, *object, nullptr, true);
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBottomOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        return mesh.topOrBottomBoundaryNodeSet(*geometry, *object, &path, false);
    } );
}

ExtrudedTriangularMesh3D::Boundary ExtrudedTriangularMesh3D::getBottomOfBoundary(shared_ptr<const GeometryObject> object) {
    return Boundary( [=](const ExtrudedTriangularMesh3D& mesh, const shared_ptr<const GeometryD<3>>& geometry) {
        return mesh.topOrBottomBoundaryNodeSet(*geometry, *object, nullptr, false);
    } );
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

    const ExtrudedTriangularMesh3D& orginal_src_mesh = this->src_mesh->getOriginalMesh();
    const MeshAxis& vertAxis = *orginal_src_mesh.vertAxis;
    if (wrapped_point.vert() < vertAxis[0] || vertAxis[vertAxis.size()] < wrapped_point.vert())
        return NaN<decltype(this->src_vec[0])>();

    const auto wrapped_longTran = to_longTran(wrapped_point);

    std::size_t longTran_element_index = this->elementIndex.getIndex(wrapped_longTran);
    if (longTran_element_index == TriangularMesh2D::ElementIndex::INDEX_NOT_FOUND)
        return NaN<decltype(this->src_vec[0])>();

    return this->flags.postprocess(point,
                                   this->src_vec[orginal_src_mesh.elementIndex(
                                        longTran_element_index,
                                        vertAxis.findUpIndex(wrapped_point.vert())-1
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
