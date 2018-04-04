#ifndef RECTANGULAR_FILTERED_H
#define RECTANGULAR_FILTERED_H

#include "rectangular_filtered_common.h"

namespace plask {

struct RectangularFilteredMesh2D: public RectangularFilteredMeshBase<2> {

    typedef std::function<bool(const typename RectangularMesh<2>::Element&)> Predicate;

    class Element {

        const RectangularFilteredMesh2D& filteredMesh;

        //std::uint32_t elementNumber;    ///< index of element in oryginal mesh
        std::size_t index0, index1; // probably this form allows to do most operation fastest in average, low indexes of element corner or just element indexes

        const RectangularMesh<2>& rectangularMesh() const { return *filteredMesh.rectangularMesh; }

    public:

        Element(const RectangularFilteredMesh2D& filteredMesh, std::size_t elementIndexOfFullMesh)
            : filteredMesh(filteredMesh)
        {
            const std::size_t v = rectangularMesh().getElementMeshLowIndex(elementIndexOfFullMesh);
            index0 = rectangularMesh().index0(v);
            index1 = rectangularMesh().index1(v);
        }

        /// \return tran index of the element
        inline std::size_t getIndex0() const { return index0; }

        /// \return vert index of the element
        inline std::size_t getIndex1() const { return index1; }

        /// \return tran index of the left edge of the element
        inline std::size_t getLowerIndex0() const { return index0; }

        /// \return vert index of the bottom edge of the element
        inline std::size_t getLowerIndex1() const { return index1; }

        /// \return tran coordinate of the left edge of the element
        inline double getLower0() const { return rectangularMesh().axis0->at(index0); }

        /// \return vert coordinate of the bottom edge of the element
        inline double getLower1() const { return rectangularMesh().axis1->at(index1); }
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
