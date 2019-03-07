#ifndef PLASK__MESH_EXTRUDED_TRIANGULAR3D_H
#define PLASK__MESH_EXTRUDED_TRIANGULAR3D_H

#include "axis1d.h"
#include "triangular2d.h"

#include <boost/icl/interval_set.hpp>
#include <boost/icl/closed_interval.hpp>

namespace plask {

/**
 * 3D mesh that is a cartesian product of 2D triangular mesh at long-tran and 1D mesh at vert axis.
 */
struct PLASK_API ExtrudedTriangularMesh3D: public MeshD<3> {

    typedef plask::Boundary<ExtrudedTriangularMesh3D> Boundary;

    TriangularMesh2D longTranMesh;

    const shared_ptr<MeshAxis> vertAxis;

    /// Iteration order, if true vert axis is changed the fastest, else it is changed the slowest.
    bool vertFastest;

    /**
     * Represent FEM-like element (right triangular prism) in ExtrudedTriangularMesh3D.
     */
    struct PLASK_API Element {
        const ExtrudedTriangularMesh3D& mesh;
        std::size_t longTranIndex, vertIndex;

        Element(const ExtrudedTriangularMesh3D& mesh, std::size_t longTranIndex, std::size_t vertIndex)
            : mesh(mesh), longTranIndex(longTranIndex), vertIndex(vertIndex) {}

        Element(const ExtrudedTriangularMesh3D& mesh, std::size_t elementIndex);

        /// @return index of this element
        std::size_t getIndex() const { return mesh.elementIndex(longTranIndex, vertIndex); }

        /**
         * Get mesh index of vertex of the bottom triangle.
         * @param bottom_triangle_node_nr index of vertex in the triangle; equals to 0, 1 or 2
         * @return mesh index of vertex of the bottom triangle
         */
        std::size_t getBottomNodeIndex(std::size_t bottom_triangle_node_nr) const {
            return mesh.index(longTranElement().getNodeIndex(bottom_triangle_node_nr), vertIndex);
        }

        /**
         * Get mesh index of vertex of the top triangle.
         * @param top_triangle_node_nr index of vertex in the triangle; equals to 0, 1 or 2
         * @return mesh index of vertex of the top triangle
         */
        std::size_t getTopNodeIndex(std::size_t top_triangle_node_nr) const {
            return mesh.index(longTranElement().getNodeIndex(top_triangle_node_nr), vertIndex+1);
        }

        /**
         * Get coordinates of the bottom base (triangle) vertex.
         * @param index index of vertex in the triangle; equals to 0, 1 or 2
         * @return coordinates of the bottom base vertex
         */
        Vec<3, double> getBottomNode(std::size_t bottom_triangle_node_nr) const {
            return mesh.at(longTranElement().getNodeIndex(bottom_triangle_node_nr), vertIndex);
        }

        /**
         * Get coordinates of the top base (triangle) vertex.
         * @param index index of vertex in the triangle; equals to 0, 1 or 2
         * @return coordinates of the top base vertex
         */
        Vec<3, double> getTopNode(std::size_t bottom_triangle_node_nr) const {
            return mesh.at(longTranElement().getNodeIndex(bottom_triangle_node_nr), vertIndex+1);
        }

        /// @return position of the middle of the element
        Vec<3, double> getMidpoint() const;

        /**
         * Get area of the prism base (which is a triangle).
         * @return the area of the prism base
         */
        double getBaseArea() const { return longTranElement().getArea(); }

        /**
         * Get height of the prism base.
         * @return the height
         */
        double getHeight() const { return mesh.vertAxis->at(vertIndex+1) - mesh.vertAxis->at(vertIndex); }

        //@{
        /**
         * Get volume of the prism represented by this element.
         * @return the volume of the element
         */
        double getArea() const { return getBaseArea() * getHeight(); }
        double getVolume() const { return getArea(); }
        //@}

        /**
         * Check if point @p p is included in @c this element.
         * @param p point to check
         * @return @c true only if @p p is included in @c this
         */
        bool includes(Vec<3, double> p) const;

        /**
         * Calculate minimal box which contains this element.
         * @return calculated box
         */
        Box3D getBoundingBox() const;

    private:
        TriangularMesh2D::Element longTranElement() const { return mesh.longTranMesh.element(longTranIndex); }
    };

    /**
     * Wrapper to ExtrudedTriangularMesh3D which allows for accessing FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    class PLASK_API Elements {

        static inline Element deref(const ExtrudedTriangularMesh3D& mesh, std::size_t index) { return mesh.getElement(index); }
    public:
        typedef IndexedIterator<const ExtrudedTriangularMesh3D, Element, deref> const_iterator;
        typedef const_iterator iterator;

        const ExtrudedTriangularMesh3D* mesh;

        explicit Elements(const ExtrudedTriangularMesh3D& mesh): mesh(&mesh) {}

        Element at(std::size_t index) const {
            if (index >= mesh->getElementsCount())
                throw OutOfBoundsException("ExtrudedTriangularMesh3D::Elements::at", "index", index, 0, mesh->getElementsCount()-1);
            return Element(*mesh, index);
        }

        Element operator[](std::size_t index) const {
            return Element(*mesh, index);
        }

        /**
         * Get number of elements (right triangular prisms) in the mesh.
         * @return number of elements
         */
        std::size_t size() const { return mesh->getElementsCount(); }

        bool empty() const { return (mesh->vertAxis->size() <= 1) || (mesh->longTranMesh.getElementsCount() == 0); }

        /// @return iterator referring to the first element
        const_iterator begin() const { return const_iterator(mesh, 0); }

        /// @return iterator referring to the past-the-end element
        const_iterator end() const { return const_iterator(mesh, size()); }
    };

    class ElementMesh;

    /**
     * Return a mesh that enables iterating over middle points of the rectangles.
     * \return the mesh
     */
    shared_ptr<ElementMesh> getElementMesh() const { return make_shared<ElementMesh>(this); }

    /// Accessor to FEM-like elements.
    Elements elements() const { return Elements(*this); }
    Elements getElements() const { return elements(); }

    //@{
    /**
     * Get an element with a given index @p elementIndex.
     * @param elementIndex index of the element
     * @return the element
     */
    Element element(std::size_t elementIndex) const { return Element(*this, elementIndex); }
    Element getElement(std::size_t elementIndex) const { return element(elementIndex); }
    //@}

    Vec<3, double> at(std::size_t index) const override;

    std::size_t size() const override;

    bool empty() const override;

    void writeXML(XMLElement& object) const override;

    /**
     * Get coordinates of node pointed by given indexes (longTranIndex and vertIndex).
     * @param longTranIndex index of longTranMesh
     * @param vertIndex index of vertAxis
     * @return the coordinates of node
     */
    Vec<3, double> at(std::size_t longTranIndex, std::size_t vertIndex) const;

    /**
     * Calculate index of this mesh using indexes of embeded meshes.
     * @param longTranIndex index of longTranMesh
     * @param vertIndex index of vertAxis
     * @return the index of this mesh
     */
    std::size_t index(std::size_t longTranIndex, std::size_t vertIndex) const {
        return vertFastest ?
            longTranIndex * vertAxis->size() + vertIndex :
            vertIndex * longTranMesh.size() + longTranIndex;
    }

    /**
     * Calculate indexes of embeded meshes from index of this mesh.
     * @param index index of this mesh
     * @return a pair: (index of longTranMesh, index of vertAxis)
     */
    std::pair<std::size_t, std::size_t> longTranAndVertIndices(std::size_t index) const;

    /**
     * Calculate element index of this mesh using element indexes of embeded meshes.
     * @param longTranIndex index of longTranMesh element
     * @param vertIndex index of vertAxis element
     * @return the element index of this mesh
     */
    std::size_t elementIndex(std::size_t longTranElementIndex, std::size_t vertElementIndex) const {
        return vertFastest ?
            longTranElementIndex * (vertAxis->size()-1) + vertElementIndex :
            vertElementIndex * longTranMesh.getElementsCount() + longTranElementIndex;
    }

    /**
     * Get number of elements in this mesh.
     * @return number of elements
     */
    std::size_t getElementsCount() const {
        const std::size_t vertSize = vertAxis->size();
        return vertSize == 0 ? 0 : (vertSize-1) * longTranMesh.getElementsCount();
    }

private:
    enum class SideBoundaryDir { BACK, FRONT, LEFT, RIGHT, ALL };

    static constexpr TriangularMesh2D::BoundaryDir boundaryDir3Dto2D(SideBoundaryDir d) { return TriangularMesh2D::BoundaryDir(d); }

    typedef boost::icl::closed_interval<std::size_t> LayersInterval;
    typedef boost::icl::interval_set<std::size_t, std::less, LayersInterval> LayersIntervalSet;

    template <SideBoundaryDir boundaryDir>
    std::set<std::size_t> boundaryNodes(const LayersIntervalSet& layers, const GeometryD<3>& geometry, const GeometryObject& object, const PathHints *path = nullptr) const;

    TriangularMesh2D::SegmentsCounts countSegmentsIn(std::size_t layer, const GeometryD<3> &geometry, const GeometryObject &object, const PathHints *path = nullptr) const;

    LayersIntervalSet layersIn(const std::vector<Box3D>& boxes) const;

    template <SideBoundaryDir boundaryDir>
    static Boundary getObjBoundary(shared_ptr<const GeometryObject> object, const PathHints &path);

    template <SideBoundaryDir boundaryDir>
    static Boundary getObjBoundary(shared_ptr<const GeometryObject> object);

    // for left, right, front, back boundaries of whole mesh or box:
    struct ExtrudedTriangularBoundaryImpl: public BoundaryNodeSetImpl {

        struct IteratorImpl: public BoundaryNodeSetImpl::IteratorImpl {

            const ExtrudedTriangularBoundaryImpl &boundary;

            std::set<std::size_t>::const_iterator longTranIter;

            std::size_t vertIndex;

            IteratorImpl(const ExtrudedTriangularBoundaryImpl &boundary, std::set<std::size_t>::const_iterator longTranIter, std::size_t vertIndex)
                : boundary(boundary), longTranIter(longTranIter), vertIndex(vertIndex)
            {}

            /*IteratorImpl(const ExtrudedTriangularBoundaryImpl &boundary, std::size_t vertIndex)
                : IteratorImpl(boundary, boundary.longTranIndices.begin(), vertIndex)
            {}*/

            std::size_t dereference() const override {
                return boundary.mesh.index(*longTranIter, vertIndex);
            }

            void increment() override {
                if (boundary.mesh.vertFastest) {
                    ++vertIndex;
                    if (vertIndex == boundary.vertIndexEnd) {
                        vertIndex = boundary.vertIndexBegin;
                        ++longTranIter;
                    }
                } else {
                    ++longTranIter;
                    if (longTranIter == boundary.longTranIndices.end()) {
                        longTranIter = boundary.longTranIndices.begin();
                        ++vertIndex;
                    }
                }
            }

            virtual bool equal(const typename BoundaryNodeSetImpl::IteratorImpl& other) const override {
                return longTranIter == static_cast<const IteratorImpl&>(other).longTranIter &&
                       vertIndex == static_cast<const IteratorImpl&>(other).vertIndex;
            }

            std::unique_ptr<PolymorphicForwardIteratorImpl<std::size_t, std::size_t>> clone() const override {
                return std::unique_ptr<PolymorphicForwardIteratorImpl<std::size_t, std::size_t>>(new IteratorImpl(*this));
            }

        };

        const ExtrudedTriangularMesh3D &mesh;

        std::set<std::size_t> longTranIndices;

        std::size_t vertIndexBegin, vertIndexEnd;

        ExtrudedTriangularBoundaryImpl(
                const ExtrudedTriangularMesh3D &mesh,
                std::set<std::size_t> longTranIndices,
                std::size_t vertIndexBegin, std::size_t vertIndexEnd)
            : mesh(mesh), longTranIndices(std::move(longTranIndices)), vertIndexBegin(vertIndexBegin), vertIndexEnd(vertIndexEnd)
        {
        }

        bool contains(std::size_t mesh_index) const override {
            std::pair<std::size_t, std::size_t> lt_v = mesh.longTranAndVertIndices(mesh_index);
            return vertIndexBegin <= lt_v.second && lt_v.second < vertIndexEnd
                    && longTranIndices.find(lt_v.first) != longTranIndices.end();
        }

        bool empty() const override { return vertIndexBegin == vertIndexEnd || longTranIndices.empty(); }

        std::size_t size() const override { return (vertIndexEnd - vertIndexBegin) * longTranIndices.size(); }

        BoundaryNodeSetImpl::const_iterator begin() const override {
            return BoundaryNodeSetImpl::const_iterator(new IteratorImpl(*this, longTranIndices.begin(), vertIndexBegin));
        }

        BoundaryNodeSetImpl::const_iterator end() const override {
            return BoundaryNodeSetImpl::const_iterator(
                mesh.vertFastest ?
                    new IteratorImpl(*this, longTranIndices.end(), vertIndexBegin) :
                    new IteratorImpl(*this, longTranIndices.begin(), vertIndexEnd)
            );
        }
    };




public:

    static Boundary getBackOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path);
    static Boundary getBackOfBoundary(shared_ptr<const GeometryObject> object);
    static Boundary getBackOfBoundary(shared_ptr<const GeometryObject> object, const PathHints *path) {
        return path ? getBackOfBoundary(object, *path) : getBackOfBoundary(object);
    }

    static Boundary getFrontOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path);
    static Boundary getFrontOfBoundary(shared_ptr<const GeometryObject> object);
    static Boundary getFrontOfBoundary(shared_ptr<const GeometryObject> object, const PathHints *path) {
        return path ? getFrontOfBoundary(object, *path) : getFrontOfBoundary(object);
    }

    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path);
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object);
    static Boundary getLeftOfBoundary(shared_ptr<const GeometryObject> object, const PathHints *path) {
        return path ? getLeftOfBoundary(object, *path) : getLeftOfBoundary(object);
    }

    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints &path);
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object);
    static Boundary getRightOfBoundary(shared_ptr<const GeometryObject> object, const PathHints *path) {
        return path ? getRightOfBoundary(object, *path) : getRightOfBoundary(object);
    }

    static Boundary getAllBoundaryIn(shared_ptr<const GeometryObject> object, const PathHints& path);
    static Boundary getAllBoundaryIn(shared_ptr<const GeometryObject> object);
    static Boundary getAllBoundaryIn(shared_ptr<const GeometryObject> object, const PathHints *path) {
        return path ? getAllBoundaryIn(object, *path) : getAllBoundaryIn(object);
    }
};

class ExtrudedTriangularMesh3D::ElementMesh: public MeshD<3> {

    /// Original mesh
    const ExtrudedTriangularMesh3D* originalMesh;

  public:
    ElementMesh(const ExtrudedTriangularMesh3D* originalMesh): originalMesh(originalMesh) {}

    LocalCoords at(std::size_t index) const override {
        return originalMesh->element(index).getMidpoint();
    }

    std::size_t size() const override {
        return originalMesh->getElementsCount();
    }

    const ExtrudedTriangularMesh3D& getOriginalMesh() const { return *originalMesh; }
};


// ------------------ Nearest Neighbor interpolation ---------------------

template <typename DstT, typename SrcT>
struct PLASK_API NearestNeighborExtrudedTriangularMesh3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D, const SrcT>
{
    RtreeOfTriangularMesh2DNodes nodesIndex;

    NearestNeighborExtrudedTriangularMesh3DLazyDataImpl(
                const shared_ptr<const ExtrudedTriangularMesh3D>& src_mesh,
                const DataVector<const SrcT>& src_vec,
                const shared_ptr<const MeshD<3>>& dst_mesh,
                const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<ExtrudedTriangularMesh3D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const ExtrudedTriangularMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags)
    {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborExtrudedTriangularMesh3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


// ------------------ Barycentric / Linear interpolation ---------------------

template <typename DstT, typename SrcT>
struct PLASK_API BarycentricExtrudedTriangularMesh3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D, const SrcT>
{
    TriangularMesh2D::ElementIndex elementIndex;

    BarycentricExtrudedTriangularMesh3DLazyDataImpl(
                const shared_ptr<const ExtrudedTriangularMesh3D>& src_mesh,
                const DataVector<const SrcT>& src_vec,
                const shared_ptr<const MeshD<3>>& dst_mesh,
                const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<ExtrudedTriangularMesh3D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const ExtrudedTriangularMesh3D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags)
    {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new BarycentricExtrudedTriangularMesh3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


// ------------------ Element mesh Nearest Neighbor interpolation ---------------------

template <typename DstT, typename SrcT>
struct PLASK_API NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, ExtrudedTriangularMesh3D::ElementMesh, const SrcT>
{
    TriangularMesh2D::ElementIndex elementIndex;

    NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl(
                const shared_ptr<const ExtrudedTriangularMesh3D::ElementMesh>& src_mesh,
                const DataVector<const SrcT>& src_vec,
                const shared_ptr<const MeshD<3>>& dst_mesh,
                const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<ExtrudedTriangularMesh3D::ElementMesh, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const ExtrudedTriangularMesh3D::ElementMesh>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags)
    {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborElementExtrudedTriangularMesh3DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                                              typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};


}   // namespace plask

#endif // PLASK__MESH_EXTRUDED_TRIANGULAR3D_H
