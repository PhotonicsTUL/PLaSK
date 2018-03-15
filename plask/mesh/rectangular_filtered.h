#ifndef RECTANGULAR_FILTERED_H
#define RECTANGULAR_FILTERED_H

#include <functional>

#include "rectangular.h"
#include "../utils/numbers_set.h"


namespace plask {

template <int DIM>
class RectangularFilteredMesh {

    const RectangularMesh<DIM>* rectangularMesh;    // TODO jaki wskaźnik? może kopia?

    CompressedSetOfNumbers<std::uint32_t> nodes;

    CompressedSetOfNumbers<std::uint32_t> elements;

public:

    typedef typename RectangularMesh<DIM>::Element FullRectElement;

    typedef std::function<bool(const FullRectElement&)> Predicate;

    RectangularFilteredMesh(const RectangularMesh<DIM>* rectangularMesh, const Predicate& predicate)
        : rectangularMesh(rectangularMesh)
    {
        for (auto el_it = rectangularMesh->elements.begin(); el_it != rectangularMesh->elements.end(); ++el_it)
            if (predicate(*el_it)) {
                // TODO wersja 3D
                elements.push_back(el_it.index);
                nodes.insert(el_it->getLoLoIndex());
                nodes.insert(el_it->getLoUpIndex());
                nodes.insert(el_it->getUpLoIndex());
                nodes.push_back(el_it->getUpUpIndex());
            }
        nodes.shrink_to_fit();
        elements.shrink_to_fit();
    }
};

}   // namespace plask

#endif // RECTANGULAR_FILTERED_H
