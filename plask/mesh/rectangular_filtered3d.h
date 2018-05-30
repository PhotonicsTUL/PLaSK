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

};

}   // namespace plask

#endif // PLASK__RECTANGULAR_FILTERED3D_H
