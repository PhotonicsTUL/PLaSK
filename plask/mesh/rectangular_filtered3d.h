#ifndef PLASK__RECTANGULAR_FILTERED3D_H
#define PLASK__RECTANGULAR_FILTERED3D_H

#include "rectangular_filtered_common.h"

namespace plask {

struct PLASK_API RectangularFilteredMesh3D: public RectangularFilteredMeshBase<3> {
    // TODO

protected:

    // Common code for: left, right, bottom, top boundries:
    template <int CHANGE_DIR_FASTER, int CHANGE_DIR_SLOWER>
    struct BoundaryIteratorImpl: public plask::BoundaryNodeSetImpl::IteratorImpl {

        const RectangularFilteredMeshBase<3> &mesh;

        /// current indexes
        Vec<3, std::size_t> index;

        /// past the last index of change direction
        const std::size_t indexFasterBegin, indexFasterEnd, indexSlowerEnd;

    private:
        /// Increase indexes without filtering.
        void naiveIncrement() {
            ++index[CHANGE_DIR_FASTER];
            if (index[CHANGE_DIR_FASTER] == indexFasterEnd) {
                index[CHANGE_DIR_FASTER] = indexFasterBegin;
                ++index[CHANGE_DIR_SLOWER];
            }
        }
    public:
        BoundaryIteratorImpl(const RectangularFilteredMeshBase<3>& mesh, Vec<3, std::size_t> index, std::size_t indexFasterEnd, std::size_t indexSlowerEnd)
            : mesh(mesh), index(index), indexFasterBegin(index[CHANGE_DIR_FASTER]), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd)
        {
            // go to the first index existed in order to make dereference possible:
            while (this->index[CHANGE_DIR_SLOWER] < indexSlowerEnd && mesh.index(this->index) == NOT_INCLUDED)
                naiveIncrement();
        }

        void increment() override {
            do {
                naiveIncrement();
            } while (index[CHANGE_DIR_SLOWER] < indexSlowerEnd && mesh.index(index) == NOT_INCLUDED);
        }

        bool equal(const plask::BoundaryNodeSetImpl::IteratorImpl& other) const override {
            return index == static_cast<const BoundaryIteratorImpl&>(other).index;
        }

        std::size_t dereference() const override {
            return mesh.index(index);
        }

        typename plask::BoundaryNodeSetImpl::IteratorImpl* clone() const override {
            return new BoundaryIteratorImpl<CHANGE_DIR_FASTER, CHANGE_DIR_SLOWER>(*this);
        }

    };

    template <int CHANGE_DIR_FASTER, int CHANGE_DIR_SLOWER>
    struct BoundaryNodeSetImpl: public BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>> {

        using typename BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>::const_iterator;

        /// first index
        Vec<3, std::size_t> index;

        /// past the last index of change directions
        std::size_t indexFasterEnd, indexSlowerEnd;

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, Vec<3, std::size_t> index, std::size_t indexFasterEnd, std::size_t indexSlowerEnd)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>(mesh), index(index), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd) {}

        BoundaryNodeSetImpl(const RectangularFilteredMeshBase<DIM>& mesh, std::size_t index0, std::size_t index1, std::size_t index2, std::size_t indexFasterEnd, std::size_t indexSlowerEnd)
            : BoundaryNodeSetWithMeshImpl<RectangularFilteredMeshBase<3>>(mesh), index(index0, index1, index2), indexFasterEnd(indexFasterEnd), indexSlowerEnd(indexSlowerEnd) {}

        bool contains(std::size_t mesh_index) const override {
            if (mesh_index >= this->mesh.size()) return false;
            Vec<3, std::size_t> mesh_indexes = this->mesh.indexes(mesh_index);
            for (int i = 0; i < 3; ++i)
                if (i == CHANGE_DIR_FASTER) {
                    if (mesh_indexes[i] < index[i] || mesh_indexes[i] >= indexFasterEnd) return false;
                } else if (i == CHANGE_DIR_SLOWER) {
                    if (mesh_indexes[i] < index[i] || mesh_indexes[i] >= indexSlowerEnd) return false;
                } else
                    if (mesh_indexes[i] != index[i]) return false;
            return true;
        }

        const_iterator begin() const override {
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR_FASTER, CHANGE_DIR_SLOWER>(this->mesh, index, indexFasterEnd, indexSlowerEnd));
        }

        const_iterator end() const override {
            Vec<3, std::size_t> index_end = index;
            index_end[CHANGE_DIR_FASTER] = indexFasterEnd;
            index_end[CHANGE_DIR_SLOWER] = indexSlowerEnd;
            return Iterator(new BoundaryIteratorImpl<CHANGE_DIR_FASTER, CHANGE_DIR_SLOWER>(this->mesh, index_end, indexFasterEnd, indexSlowerEnd));
        }
    };

    public:     // boundaries:

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0,
                                                             std::size_t index1Begin, std::size_t index1End,
                                                             std::size_t index2Begin, std::size_t index2End
                                                             ) const override
    {
        return new BoundaryNodeSetImpl<1, 2>(*this, line_nr_axis0, index1Begin, index1End, index2Begin, index2End);
    }

    BoundaryNodeSet createIndex0BoundaryAtLine(std::size_t line_nr_axis0) const override {
        return createIndex0BoundaryAtLine(line_nr_axis0, boundaryIndex[1].lo, boundaryIndex[1].up, boundaryIndex[2].lo, boundaryIndex[2].up);
    }

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1,
                                                             std::size_t index0Begin, std::size_t index0End,
                                                             std::size_t index2Begin, std::size_t index2End
                                                             ) const override
    {
        return new BoundaryNodeSetImpl<0, 2>(*this, line_nr_axis1, index0Begin, index0End, index2Begin, index2End);
    }

    BoundaryNodeSet createIndex1BoundaryAtLine(std::size_t line_nr_axis1) const override {
        return createIndex1BoundaryAtLine(line_nr_axis1, boundaryIndex[0].lo, boundaryIndex[0].up, boundaryIndex[2].lo, boundaryIndex[2].up);
    }

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2,
                                                             std::size_t index0Begin, std::size_t index0End,
                                                             std::size_t index1Begin, std::size_t index1End
                                                             ) const override
    {
        return new BoundaryNodeSetImpl<0, 1>(*this, line_nr_axis2, index0Begin, index0End, index1Begin, index1End);
    }

    BoundaryNodeSet createIndex2BoundaryAtLine(std::size_t line_nr_axis2) const override {
        return createIndex2BoundaryAtLine(line_nr_axis2, boundaryIndex[0].lo, boundaryIndex[0].up, boundaryIndex[1].lo, boundaryIndex[1].up);
    }

    BoundaryNodeSet createBackBoundary() const override {
        return createIndex0BoundaryAtLine(boundaryIndex[0].lo);
    }

    BoundaryNodeSet createFrontBoundary() const override {
        return createIndex0BoundaryAtLine(boundaryIndex[0].up);
    }

    BoundaryNodeSet createLeftBoundary() const override {
        return createIndex1BoundaryAtLine(boundaryIndex[1].lo);
    }

    BoundaryNodeSet createRightBoundary() const override {
        return createIndex1BoundaryAtLine(boundaryIndex[1].up);
    }

    BoundaryNodeSet createBottomBoundary() const override {
        return createIndex2BoundaryAtLine(boundaryIndex[2].lo);
    }

    BoundaryNodeSet createTopBoundary() const override {
        return createIndex2BoundaryAtLine(boundaryIndex[2].up);
    }
};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED3D_H
