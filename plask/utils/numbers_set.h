#ifndef PLASK__UTILS_NUMBERS_SET_H
#define PLASK__UTILS_NUMBERS_SET_H

#include <vector>
#include <algorithm>
#include <limits>
#include "../exceptions.h"

namespace plask {

/**
 * Sorted, compressed, indexed set of numbers.
 */
template <typename number_t = std::size_t>
class CompressedSetOfNumbers {

    struct Segment {

        number_t numberEnd;     ///< last number in the segment increased by one

        number_t indexEnd;      ///< accumulated count of segments length, up to this and including this = index-1 of numberEnd-1 in the set

        bool operator<(const number_t index) const { return this->indexEnd < index; }

    };

    std::vector<Segment> segments;

    number_t sizeOfSegment(std::size_t seg_nr) {
        return (seg_nr == 0) ? segments.front().indexEnd : (segments[seg_nr].indexEnd - segments[seg_nr-1].indexEnd);
    }

    number_t sizeOfSegment(std::vector<Segment>::const_iterator it) {
        return (it == segments.begin()) ? it->indexEnd : (it->indexEnd - (it-1)->indexEnd);
    }

public:

    /**
     * Get number of items (numbers) included in the set.
     * @return number of numbers included in the set
     */
    std::size_t size() const { return segments.empty() ? 0 : segments.back().indexEnd; }

    /**
     * Check if the set is empty.
     * @return @c true only if the set is empty
     */
    bool empty() const { return segments.empty(); }

    /**
     * Get number at a given @p index in the set, without checking if the @p index is valid.
     * @param index index in the set, must be in range [0, size())
     * @return number at @p index
     */
    number_t operator[](std::size_t index) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), index);
        // here: index < seg_it->indexEnd
        return seg_it->numberEnd + index - seg_it->indexEnd;
    }

    /**
     * Get number at a given @p index or throw exception if the @p index is not valid.
     * @param index index in the set, it is valid if it is included in range [0, size())
     * @return number at @p index
     */
    number_t at(std::size_t index) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), index);
        if (seg_it == segments.end()) throw OutOfBoundsException("CompressedSetOfNumbers::at", "index", index, 0, this->size());
        // here: index < seg_it->indexEnd
        return seg_it->numberEnd + index - seg_it->indexEnd;
    }

    /// Constant returned by indexOf method for numbers not included in the set.
    constexpr std::size_t NOT_INCLUDED = std::numeric_limits<std::size_t>::max();

    /**
     * Get index of a given @p number in the set.
     * @param number item to find
     * @return either index of @p number in the set or @c NOT_INCLUDED if @p number is not included in the set
     */
    std::size_t indexOf(number_t number) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), number,
                [](const Segment& seg, number_t n) { return seg.numberEnd < n; });
        if (seg_it == segments.end()) return NOT_INCLUDED;  // number is too big
        // TODO
        // here: number < seg_it->numberEnd
        //seg_it->indexEnd + number - seg_it->numberEnd;
    }

    void insert(number_t number) {
        // TODO
    }

};

}   // namespace plask

#endif // PLASK__UTILS_NUMBERS_SET_H
