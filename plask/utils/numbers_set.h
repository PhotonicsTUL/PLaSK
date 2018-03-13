#ifndef PLASK__UTILS_NUMBERS_SET_H
#define PLASK__UTILS_NUMBERS_SET_H

#include <vector>
#include <algorithm>

namespace plask {

/**
 * Sorted, compressed, indexed set of numbers.
 */
template <typename number_t = std::size_t>
class CompressedSetOfNumbers {

    struct Segment {

        number_t first;          ///< first number in the segment //TODO -> last??

        number_t accumCount;     ///< accumulated count of segments length, up to this = position of the last number in this segment - 1

        bool operator<(const number_t accumCount) const { return this->accumCount < accumCount; }

    };

    std::vector<Segment> segments;

    number_t sizeOfSegment(std::size_t seg_nr) {
        return (seg_nr == 0) ? segments.front().accumCount : (segments[seg_nr].accumCount - segments[seg_nr-1].accumCount);
    }

    number_t sizeOfSegment(std::vector<Segment>::const_iterator it) {
        return (it == segments.begin()) ? it->accomCount : (it->accomCount - (it-1)->accomCount);
    }

public:

    /**
     * Get number of items (numbers) included in the set.
     * @return number of numbers included in the set
     */
    std::size_t size() const { return segments.empty() ? 0 : segments.back().accumCount; }

    /**
     * Check if the set is empty.
     * @return @c true only if the set is empty
     */
    bool empty() const { return segments.empty(); }

    number_t operator[](std::size_t index) const {  // TODO wrong
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), index);
        // here: index < seg_it->accumCount
        return seg_it->first + seg_it->accumCount - index;
    }

};

}   // namespace plask

#endif // PLASK__UTILS_NUMBERS_SET_H
