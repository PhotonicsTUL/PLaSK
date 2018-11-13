#ifndef PLASK__TRIANGULAR2D_H
#define PLASK__TRIANGULAR2D_H

#include "mesh.h"
#include "interpolation.h"
#include <array>

#include <boost/geometry/index/rtree.hpp>

#include <boost/geometry/geometries/register/point.hpp>
BOOST_GEOMETRY_REGISTER_POINT_2D(plask::Vec<2>, double, cs::cartesian, c0, c1)

#include <boost/geometry/geometries/box.hpp>

namespace plask {

struct TriangularMesh2D: public MeshD<2> {

    using MeshD<2>::LocalCoords;
    typedef std::vector<LocalCoords> LocalCoordsVec;
    typedef LocalCoordsVec::const_iterator const_iterator;
    typedef const_iterator iterator;

    LocalCoordsVec nodes;

    typedef std::array<std::size_t, 3> TriangleNodeIndexes;

    std::vector< TriangleNodeIndexes > elementNodes;

    /**
     * Represent FEM-like element (triangle) in TriangularMesh2D.
     */
    struct Element {
        TriangleNodeIndexes triangleNodes;
        const TriangularMesh2D& mesh;   // for getting access to the nodes

        Element(const TriangularMesh2D& mesh, TriangleNodeIndexes triangleNodes)
            : triangleNodes(triangleNodes), mesh(mesh) {}

        /**
         * Get index of the triangle vertex in mesh (nodes vector).
         * @param index index of vertex in the triangle corresponded to this element; equals to 0, 1 or 2
         * @return index of the triangle vertex in mesh (nodes vector)
         */
        std::size_t getNodeIndex(std::size_t index) const noexcept {
            assert(index < 3);
            return triangleNodes[index];
        }

        /**
         * Get coordinate of the triangle vertex.
         * @param index index of vertex in the triangle corresponded to this element; equals to 0, 1 or 2
         * @return coordinate of the triangle vertex
         */
        const LocalCoords& getNode(std::size_t index) const noexcept {
            return mesh.nodes[getNodeIndex(index)];
        }

        /**
         * Get coordinates of the triangle vertices.
         * @return coordinates of the triangle vertices
         */
        std::array<LocalCoords, 3> getNodes() const {
            return { getNode(0), getNode(1), getNode(2) };
        }

        /**
         * Get centroid of the triangle corresponded to this element.
         * @return centroid of the triangle corresponded to this element
         */
        LocalCoords getMidpoint() const {
            return (getNode(0)+getNode(1)+getNode(2)) / 3.0;
        }

        /**
         * Get area of the triangle represented by this element.
         * @return the area of the triangle
         */
        double getArea() const noexcept {
            // formula comes from http://www.mathguru.com/level2/application-of-coordinate-geometry-2007101600011139.aspx
            const LocalCoords A = getNode(0);
            const LocalCoords B = getNode(1);
            const LocalCoords C = getNode(2);
            return abs( (A.c0 - C.c0) * (B.c1 - A.c1)
                      - (A.c0 - B.c0) * (C.c1 - A.c1) ) / 2.0;
        }

        /**
         * Calculate barycentric (area) coordinates of the point @p p with respect to the triangle represented by this.
         * @param p point
         * @return the barycentric (area) coordinates of @c p
         */
        Vec<3, double> barycentric(Vec<2, double> p) const;

        /**
         * Calculate minimal rectangle which contains the triangle represented by the element.
         * @return calculated rectangle
         */
        Box2D getBoundingBox() const;
    };

    /**
     * Wrapper to TriangularMesh2D which allows for accessing FEM-like elements.
     *
     * It works like read-only, random access container of @ref Element objects.
     */
    struct Elements {
        const TriangularMesh2D& mesh;

        explicit Elements(const TriangularMesh2D& mesh): mesh(mesh) {}

        Element operator[](std::size_t index) const {
            return Element(mesh, mesh.elementNodes[index]);
        }

        /**
         * Get number of elements (triangles) in the mesh.
         * @return number of elements
         */
        std::size_t size() const { return mesh.getElementsCount(); }

        typedef IndexedIterator<const Elements, Element> const_iterator;
        typedef const_iterator iterator;

        /// @return iterator referring to the first element (triangle) in the mesh
        const_iterator begin() const { return const_iterator(this, 0); }

        /// @return iterator referring to the past-the-end element (triangle) in the mesh
        const_iterator end() const { return const_iterator(this, this->size()); }
    };

    Elements getElements() const { return Elements(*this); }
    Elements elements() const { return Elements(*this); }

    Element getElement(std::size_t elementIndex) const {
        return Element(*this, elementNodes[elementIndex]);
    };

    Element element(std::size_t elementIndex) const {
        return Element(*this, elementNodes[elementIndex]);
    };

    /**
     * Get number of elements (triangles) in this mesh.
     * @return number of elements
     */
    std::size_t getElementsCount() const {
        return elementNodes.size();
    }

    /**
     * Instance of this class allows for adding triangles to the mesh effectively.
     */
    struct Builder {
        std::map<LocalCoords, std::size_t> indexOfNode; ///< map nodes to their indexes in mesh.nodes vector
        TriangularMesh2D& mesh; ///< destination mesh

        /**
         * Construct builder which will add triangles to the given @p mesh.
         * @param mesh triangles destination
         */
        explicit Builder(TriangularMesh2D& mesh);

        /**
         * Construct builder which will add triangles to the given @p mesh.
         *
         * This constructor preallocate extra space for elements and nodes in @p mesh,
         * which usually improves performance.
         * @param mesh triangles destination
         * @param predicted_number_of_elements predicted (maximal) number of elements (triangles) to be added
         * @param predicted_number_of_elements predicted (maximal) number of nodes to be added
         */
        explicit Builder(TriangularMesh2D& mesh, std::size_t predicted_number_of_elements, std::size_t predicted_number_of_nodes);

        /**
         * Construct builder which will add triangles to the given @p mesh.
         *
         * This constructor preallocate extra space for elements and nodes (3*predicted_number_of_elements) in @p mesh,
         * which usually improves performance.
         * @param mesh triangles destination
         * @param predicted_number_of_elements predicted (maximal) number of elements to be added
         */
        explicit Builder(TriangularMesh2D& mesh, std::size_t predicted_number_of_elements)
            : Builder(mesh, predicted_number_of_elements, predicted_number_of_elements*3) {}

        /// Shrink to fit both: elements and nodes vectors of destination mesh.
        ~Builder();

        /**
         * Add a triangle to the mesh.
         * @param p1, p2, p3 coordinates of vertices of the triangle to add
         * @return <code>*this</code>
         */
        Builder& add(LocalCoords p1, LocalCoords p2, LocalCoords p3);

        /**
         * Add a triangle to the mesh.
         * @param e a trianglular element (from another mesh) which defines the triangle to add
         * @return <code>*this</code>
         */
        Builder& add(const Element& e) { return add(e.getNode(0), e.getNode(1), e.getNode(2)); }

    private:

        /**
         * Add a @p node to nodes vector of destination mesh if it is not already there.
         * @param node coordinates of node to (conditionally) add
         * @return an index of the node (added or found) in the vector
         */
        std::size_t addNode(LocalCoords node);
    };

    // ------------------ Mesh and MeshD<2> interfaces: ------------------
    LocalCoords at(std::size_t index) const override {
        assert(index < nodes.size());
        return nodes[index];
    }

    std::size_t size() const override {
        return nodes.size();
    }

    bool empty() const override {
        return nodes.empty();
    }

    // ---- Faster iterators used when exact type of mesh is known; they hide polimorphic iterators of parent class ----
    const_iterator begin() const { return nodes.begin(); }
    const_iterator end() const { return nodes.end(); }

    // ----------------------- Masked meshes -----------------------

    /// Type of predicate function which returns bool for given element of a mesh.
    typedef std::function<bool(const Element&)> Predicate;

    /**
     * Construct masked mesh with elements of @c this chosen by a @p predicate.
     * Preserve order of elements of @p this.
     * @param predicate predicate which returns either @c true for accepting element or @c false for rejecting it
     * @return the masked mesh constructed
     */
    TriangularMesh2D masked(const Predicate& predicate) const;

    /**
     * Construct masked mesh with all elements of @c this which have required materials in the midpoints.
     * Preserve order of elements of @c this.
     * @param geom geometry to get materials from
     * @param materialPredicate predicate which returns either @c true for accepting material or @c false for rejecting it
     * @return the masked mesh constructed
     */
    TriangularMesh2D masked(const GeometryD<2>& geom, const std::function<bool(shared_ptr<const Material>)> materialPredicate) const {
        return masked([&](const Element& el) { return materialPredicate(geom.getMaterial(el.getMidpoint())); });
    }

    /**
     * Construct masked mesh with all elements of @c this which have required kinds of materials (in the midpoints).
     * Preserve order of elements of @p this.
     * @param geom geometry to get materials from
     * @param materialKinds one or more kinds of material encoded with bit @c or operation,
     *        e.g. @c DIELECTRIC|METAL for selecting all dielectrics and metals,
     *        or @c ~(DIELECTRIC|METAL) for selecting everything else
     * @return the masked mesh constructed
     */
    TriangularMesh2D masked(const GeometryD<2>& geom, unsigned int materialKinds) const {
        return masked([&](const Element& el) { return (geom.getMaterial(el.getMidpoint())->kind() & materialKinds) != 0; });
    }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    void writeXML(XMLElement& object) const override;
};


// ------------------ Nearest Neighbor interpolation ---------------------

template <typename DstT, typename SrcT>
struct PLASK_API NearestNeighborTriangularMesh2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>
{
    struct TriangularMesh2DGetter {
        typedef Vec<2, double> result_type;

        shared_ptr<const TriangularMesh2D> src_mesh;

        TriangularMesh2DGetter(const shared_ptr<const TriangularMesh2D>& src_mesh): src_mesh(src_mesh) {}

        result_type operator()(std::size_t index) const {
            return src_mesh->at(index);
        }
    };

    typedef boost::geometry::index::rtree<
            std::size_t,
            boost::geometry::index::quadratic<16>, //??
            TriangularMesh2DGetter
            > Rtree;

    Rtree nodesIndex;

    NearestNeighborTriangularMesh2DLazyDataImpl(
                const shared_ptr<const TriangularMesh2D>& src_mesh,
                const DataVector<const SrcT>& src_vec,
                const shared_ptr<const MeshD<2>>& dst_mesh,
                const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<TriangularMesh2D, SrcT, DstT, INTERPOLATION_NEAREST> {
    static LazyData<DstT> interpolate(const shared_ptr<const TriangularMesh2D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags)
    {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new NearestNeighborTriangularMesh2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};



// ------------------ Barycentric / Linear interpolation ---------------------

template <typename DstT, typename SrcT>
struct PLASK_API BarycentricTriangularMesh2DLazyDataImpl: public InterpolatedLazyDataImpl<DstT, TriangularMesh2D, const SrcT>
{
    typedef boost::geometry::model::box<Vec<2>> Box;

    typedef boost::geometry::index::rtree<
            std::pair<Box, std::size_t>,
            boost::geometry::index::quadratic<16> //??
            > Rtree;

    Rtree trianglesIndex;

    struct ValueGetter {
        Rtree::value_type operator()(std::size_t index) const {
            TriangularMesh2D::Element el = this->src_mesh->getElement(index);
            const auto n0 = el.getNode(0);
            const auto n1 = el.getNode(1);
            const auto n2 = el.getNode(2);
            return std::make_pair(
                        Box(
                            vec(std::min(std::min(n0.c0, n1.c0), n2.c0), std::min(std::min(n0.c1, n1.c1), n2.c1)),
                            vec(std::max(std::max(n0.c0, n1.c0), n2.c0), std::max(std::max(n0.c1, n1.c1), n2.c1))
                        ),
                        index);
        }

        shared_ptr<const TriangularMesh2D> src_mesh;

        ValueGetter(const shared_ptr<const TriangularMesh2D>& src_mesh): src_mesh(src_mesh) {}
    };

    BarycentricTriangularMesh2DLazyDataImpl(
                const shared_ptr<const TriangularMesh2D>& src_mesh,
                const DataVector<const SrcT>& src_vec,
                const shared_ptr<const MeshD<2>>& dst_mesh,
                const InterpolationFlags& flags);

    DstT at(std::size_t index) const override;
};

template <typename SrcT, typename DstT>
struct InterpolationAlgorithm<TriangularMesh2D, SrcT, DstT, INTERPOLATION_LINEAR> {
    static LazyData<DstT> interpolate(const shared_ptr<const TriangularMesh2D>& src_mesh,
                                      const DataVector<const SrcT>& src_vec,
                                      const shared_ptr<const MeshD<3>>& dst_mesh,
                                      const InterpolationFlags& flags)
    {
        if (src_mesh->empty()) throw BadMesh("interpolate", "Source mesh empty");
        return new BarycentricTriangularMesh2DLazyDataImpl<typename std::remove_const<DstT>::type,
                                                       typename std::remove_const<SrcT>::type>
            (src_mesh, src_vec, dst_mesh, flags);
    }

};

}   // namespace plask

#endif // PLASK__TRIANGULAR2D_H
