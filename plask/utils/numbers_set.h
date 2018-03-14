#ifndef PLASK__UTILS_NUMBERS_SET_H
#define PLASK__UTILS_NUMBERS_SET_H

#include <vector>
#include <algorithm>
#include <limits>
#include "../exceptions.h"

namespace plask {

/**
 * Sorted, compressed, indexed set of numbers.
 *
 * Set is stored as a sorted vector of segments, each represents a sequence of successive numbers.
 */
template <typename number_t = std::size_t>
class CompressedSetOfNumbers {

    struct Segment {

        number_t numberEnd;     ///< last number in the segment increased by one

        number_t indexEnd;      ///< accumulated count of segments length, up to this and including this = index-1 of numberEnd-1 in the set = first index in the next segment

        static bool compareByIndexEnd(number_t i, const Segment& seg) { return i < seg.indexEnd; }

        static bool compareByNumberEnd(number_t n, const Segment& seg) { return n < seg.numberEnd; }

        Segment(number_t numberEnd, number_t indexEnd): numberEnd(numberEnd), indexEnd(indexEnd) {}

    };

    std::vector<Segment> segments;

    /*number_t sizeOfSegment(std::size_t seg_nr) const {
        return (seg_nr == 0) ? segments.front().indexEnd : (segments[seg_nr].indexEnd - segments[seg_nr-1].indexEnd);
    }

    static number_t sizeOfSegment(std::vector<Segment>::const_iterator it) {
        return (it == segments.begin()) ? it->indexEnd : (it->indexEnd - (it-1)->indexEnd);
    }*/

    /**
     * Get first index in the segment pointed by @p it.
     * @param it iterator to segment
     * @return first index in the segment @c *it
     */
    number_t firstIndex(typename std::vector<Segment>::const_iterator it) const {
        return (it == segments.begin()) ? 0 : (it-1)->indexEnd;
    }

public:

    /**
     * Get number of items (numbers) included in the set.
     *
     * Time complexity: constant.
     * @return number of numbers included in the set
     */
    std::size_t size() const { return segments.empty() ? 0 : segments.back().indexEnd; }

    /**
     * Get number of segments, which is usable only for informative purposes and tests.
     *
     * Time complexity: constant.
     * @return number of segments
     */
    std::size_t segmentsCount() const { return segments.size(); }

    /**
     * Check if the set is empty.
     *
     * Time complexity: constant.
     * @return @c true only if the set is empty
     */
    bool empty() const { return segments.empty(); }

    /**
     * Get number at a given @p index in the set, without checking if the @p index is valid.
     *
     * Time complexity: logarithmic in number of segments (which never exceed the size of the set).
     * @param index index in the set, must be in range [0, size())
     * @return number at @p index
     */
    number_t operator[](std::size_t index) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), index, Segment::compareByIndexEnd);
        // here: index < seg_it->indexEnd
        assert(seg_it != segments.end());   // true for valid index
        return seg_it->numberEnd - seg_it->indexEnd + index;    // must be non-negative as numberEnd >= indexEnd
    }

    /**
     * Get number at a given @p index or throw exception if the @p index is not valid.
     *
     * Time complexity: logarithmic in number of segments (which never exceed the size of the set).
     * @param index index in the set, it is valid if it is included in range [0, size())
     * @return number at @p index
     */
    number_t at(std::size_t index) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), index, Segment::compareByIndexEnd);
        if (seg_it == segments.end()) throw OutOfBoundsException("CompressedSetOfNumbers::at", "index", index, 0, this->size());
        // here: index < seg_it->indexEnd
        return seg_it->numberEnd + index - seg_it->indexEnd;
    }

    /// Constant returned by indexOf method for numbers not included in the set.
    enum:std::size_t { NOT_INCLUDED = std::numeric_limits<std::size_t>::max() };

    /**
     * Get index of a given @p number in the set.
     *
     * Time complexity: logarithmic in number of segments (which never exceed the size of the set).
     * @param number item to find
     * @return either index of @p number in the set or @c NOT_INCLUDED if it is not included
     */
    std::size_t indexOf(number_t number) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), number, Segment::compareByNumberEnd);
        if (seg_it == segments.end()) return NOT_INCLUDED;  // number is too large
        // here: number < seg_it->numberEnd
        std::ptrdiff_t index = std::ptrdiff_t(seg_it->indexEnd) + std::ptrdiff_t(number) - std::ptrdiff_t(seg_it->numberEnd);
        // index can even be negative here
        return index >= std::ptrdiff_t(firstIndex(seg_it)) ? std::size_t(index) : NOT_INCLUDED;
    }

    /**
     * Quickly append number to the end of the set.
     *
     * Time complexity: amortized constant.
     * @param number number to add, must be larger than all numbers already included in the set
     */
    void push_back(number_t number) {
        if (empty()) {
            segments.emplace_back(number+1, 1);
        } else if (segments.back().numberEnd == number) {
            ++(segments.back().numberEnd);
            ++(segments.back().indexEnd);
        } else
            segments.emplace_back(number+1, segments.back().indexEnd+1);
    }

    /**
     * Insert @p number to the set.
     *
     * Time complexity: logarithmic (optimistic, e.g. if number is already in the set or is inserted near the end) or linear (pesymistic) in number of segments.
     * @param number number to insert
     */
    void insert(number_t number) {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), number, Segment::compareByNumberEnd);
        if (seg_it == segments.end()) { // number is larger than all numbers in the set
            push_back(number);
        } else {    // here: number < seg_it->numberEnd:
            if (seg_it == segments.begin()) {
                const number_t firstNumber = seg_it->numberEnd - seg_it->indexEnd;
                if (number >= firstNumber) return;  // already included
                // here: 0 <= number < firstNumber and we will insert number with new index
                for (auto it = seg_it; it != segments.end(); ++it) ++(it->indexEnd);
                if (number+1 == firstNumber) return;    // first segment has been enlarged by the new first element (indexEnd is already increased)
                segments.emplace(seg_it, number+1, 1);  // we can't enlarge first segment, so we need new one
            } else {
                auto prev_it = seg_it - 1;
                const number_t firstNumber = seg_it->numberEnd + prev_it->indexEnd - seg_it->indexEnd;
                if (number >= firstNumber) return;  // already included
                // we will insert number with new index
                for (auto it = seg_it; it != segments.end(); ++it) ++(it->indexEnd);
                if (number+1 == firstNumber) {          // segment pointed by seg_it has been enlarged by new first element
                    if (prev_it->numberEnd == number)  // if there was one element gap, and now it is no gap after the previous segment
                        segments.erase(prev_it);        // we have to remove the previous segment
                    return;
                }
                // here: we can't enlarge seg_it segment
                if (prev_it->numberEnd == number) {    // we can append new element to the end of the previous segment (there is still a gap after it, number+1 is in the gap)
                    ++(prev_it->numberEnd);
                    ++(prev_it->indexEnd);
                } else      // we have to insert new segment
                    segments.emplace(seg_it, number+1, prev_it->indexEnd+1);
            }
        }

    }

};

}   // namespace plask

#endif // PLASK__UTILS_NUMBERS_SET_H
