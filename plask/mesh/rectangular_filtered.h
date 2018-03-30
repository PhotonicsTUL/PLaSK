#ifndef RECTANGULAR_FILTERED_H
#define RECTANGULAR_FILTERED_H

#include <functional>

#include "rectangular.h"
#include "../utils/numbers_set.h"


namespace plask {

/**
 * Common base class for RectangularFilteredMesh 2D and 3D.
 *
 * Do not use directly.
 */
template <int DIM>
class RectangularFilteredMeshBase: public MeshD<DIM> {

protected:

    const RectangularMesh<DIM>* rectangularMesh;

    typedef CompressedSetOfNumbers<std::uint32_t> Set;

    /// numbers of rectangularMesh indexes which are in the corners of the elements enabled
    Set nodes;

    /// numbers of enabled elements
    Set elements;

    /**
     * Used by interpolation.
     * @param axis
     * @param wrapped_point_coord
     * @param index_lo
     * @param index_hi
     */
    static void findIndexes(const MeshAxis& axis, double wrapped_point_coord, std::size_t& index_lo, std::size_t& index_hi) {
        index_hi = axis.findUpIndex(wrapped_point_coord);
        if (index_hi+1 == axis.size()) --index_hi;    // p.c0 == axis0->at(axis0->size()-1)
        assert(index_hi > 0);
        index_lo = index_hi - 1;
    }

    /**
     * Used by nearest neighbor interpolation.
     * @param p point coordinate such that axis.at(index_lo) <= p <= axis.at(index_hi)
     * @param axis
     * @param index_lo, index_hi indexes
     * @return either @p index_lo or @p index_hi, index which minimize |p - axis.at(index)|
     */
    static std::size_t nearest(double p, const MeshAxis& axis, std::size_t index_lo, std::size_t index_hi) {
        return p - axis.at(index_lo) <= axis.at(index_hi) - p ? index_lo : index_hi;
    }

public:

    using typename MeshD<DIM>::LocalCoords;

    /// Returned by some methods to signalize that element or node (with given index(es)) is not included in the mesh.
    enum:std::size_t { NOT_INCLUDED = Set::NOT_INCLUDED };

    RectangularFilteredMeshBase(const RectangularMesh<DIM>* rectangularMesh)
        : rectangularMesh(rectangularMesh) {}

    /**
     * Iterator over nodes coordinates.
     *
     * Iterator of this type is faster than IndexedIterator used by parent class,
     * as it has constant time dereference operation while at method has logarithmic time complexity.
     *
     * One can use:
     * - getIndex() method of the iterator to get index of the node,
     * - getNumber() method of the iterator to get index of the node in the wrapped mesh.
     */
    class const_iterator: public Set::ConstIteratorFacade<const_iterator, LocalCoords> {

        const RectangularFilteredMeshBase* mesh;

        LocalCoords dereference() const {
            return mesh->at(this->getNumber());
        }

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const RectangularFilteredMeshBase* mesh, CtorArgs&&... ctorArgs)
            : mesh(&mesh), Set::ConstIteratorFacade<const_iterator, LocalCoords>(std::forward<CtorArgs>(ctorArgs)...) {}

        const Set& set() const { return mesh->nodes; }
    };

    /// Iterator over nodes coordinates. The same as const_iterator, since non-const iterators are not supported.
    typedef const_iterator iterator;

    const_iterator begin() const { return const_iterator(*this, 0, nodes.segments.begin()); }
    const_iterator end() const { return const_iterator(*this, size(), nodes.segments.end()); }

    LocalCoords at(std::size_t index) const override {
        return rectangularMesh->at(nodes.at(index));
    }

    std::size_t size() const override { return nodes.size(); }

    bool empty() const override { return nodes.empty(); }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
        return rectangularMesh->index0(nodes.at(mesh_index));
    }

    /**
     * Calculate index of axis0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of axis0, from 0 to axis0->size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return rectangularMesh->index1(nodes.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t majorIndex(std::size_t mesh_index) const {
        return rectangularMesh->majorIndex(nodes.at(mesh_index));
    }

    /**
     * Calculate index of major axis using given mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of major axis, from 0 to majorAxis.size()-1
     */
    inline std::size_t minorIndex(std::size_t mesh_index) const {
        return rectangularMesh->minorIndex(nodes.at(mesh_index));
    }

};

struct RectangularFilteredMesh2D: public RectangularFilteredMeshBase<2> {

    typedef std::function<bool(const typename RectangularMesh<2>::Element&)> Predicate;

    struct Element {
        // TODO
    };

    RectangularFilteredMesh2D(const RectangularMesh<2>* rectangularMesh, const Predicate& predicate)
        : RectangularFilteredMeshBase(rectangularMesh)
    {
        for (auto el_it = rectangularMesh->elements.begin(); el_it != rectangularMesh->elements.end(); ++el_it)
            if (predicate(*el_it)) {
                elements.push_back(el_it.index);
                nodes.insert(el_it->getLoLoIndex());
                nodes.insert(el_it->getLoUpIndex());
                nodes.insert(el_it->getUpLoIndex());
                nodes.push_back(el_it->getUpUpIndex());
            }
        nodes.shrink_to_fit();
        elements.shrink_to_fit();
    }

    /**
     * Calculate this mesh index using indexes of axis0 and axis1.
     * @param axis0_index index of axis0, from 0 to axis0->size()-1
     * @param axis1_index index of axis1, from 0 to axis1->size()-1
     * @return this mesh index, from 0 to size()-1, or NOT_INCLUDED
     */
    inline std::size_t index(std::size_t axis0_index, std::size_t axis1_index) const {
        return nodes.indexOf(rectangularMesh->index(axis0_index, axis1_index));
    }

    /**
     * Get point with given mesh indices.
     * @param index0 index of point in axis0
     * @param index1 index of point in axis1
     * @return point with given @p index
     */
    inline Vec<2, double> at(std::size_t index0, std::size_t index1) const {
        return rectangularMesh->at(index0, index1);
    }

    /**
     * Get point with given x and y indexes.
     * @param axis0_index index of axis0, from 0 to axis0->size()-1
     * @param axis1_index index of axis1, from 0 to axis1->size()-1
     * @return point with given axis0 and axis1 indexes
     */
    inline Vec<2,double> operator()(std::size_t axis0_index, std::size_t axis1_index) const {
        return rectangularMesh->operator()(axis0_index, axis1_index);
    }

private:
    bool canBeIncluded(const Vec<2>& point) const {
        return
            rectangularMesh->axis0->at(0) <= point[0] && point[0] <= rectangularMesh->axis0->at(rectangularMesh->axis0->size()-1) &&
            rectangularMesh->axis1->at(0) <= point[1] && point[1] <= rectangularMesh->axis1->at(rectangularMesh->axis1->size()-1);
    }

    bool prepareInterpolation(const Vec<2>& point, Vec<2>& wrapped_point, std::size_t& index0_lo, std::size_t& index0_hi, std::size_t& index1_lo, std::size_t& index1_hi, std::size_t& rectmesh_index_lo, const InterpolationFlags& flags) const {
        wrapped_point = flags.wrap(point);

        if (!canBeIncluded(wrapped_point)) return false;

        findIndexes(*rectangularMesh->axis0, wrapped_point.c0, index0_lo, index0_hi);
        findIndexes(*rectangularMesh->axis1, wrapped_point.c1, index1_lo, index1_hi);

        rectmesh_index_lo = rectangularMesh->index(index0_lo, index1_lo);
        return elements.includes(rectangularMesh->getElementIndexFromLowIndex(rectmesh_index_lo));
    }

public:
    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<2> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 interpolation::bilinear(
                                     rectangularMesh->axis0->at(index0_lo), rectangularMesh->axis0->at(index0_hi),
                                     rectangularMesh->axis1->at(index1_lo), rectangularMesh->axis1->at(index1_hi),
                                     data[nodes.indexOf(rectmesh_index_lo)],
                                     data[index(index0_hi, index1_lo)],
                                     data[index(index0_hi, index1_hi)],
                                     data[index(index0_lo, index1_hi)],
                                     wrapped_point.c0, wrapped_point.c1));
    }

    /**
     * Calculate (using nearest neighbor interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateNearestNeighbor(const RandomAccessContainer& data, const Vec<2>& point, const InterpolationFlags& flags) const
        -> typename std::remove_reference<decltype(data[0])>::type
    {
        Vec<2> wrapped_point;
        std::size_t index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo;

        if (!prepareInterpolation(point, wrapped_point, index0_lo, index0_hi, index1_lo, index1_hi, rectmesh_index_lo, flags))
            return NaNfor<decltype(data[0])>();

        return flags.postprocess(point,
                                 data[this->index(
                                     nearest(wrapped_point.c0, *rectangularMesh->axis0, index0_lo, index0_hi),
                                     nearest(wrapped_point.c1, *rectangularMesh->axis1, index1_lo, index1_hi)
                                 )]);
    }


};

}   // namespace plask

#endif // RECTANGULAR_FILTERED_H
