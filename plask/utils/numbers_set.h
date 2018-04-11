#ifndef PLASK__UTILS_NUMBERS_SET_H
#define PLASK__UTILS_NUMBERS_SET_H

#include <vector>
#include <algorithm>
#include <limits>
#include <boost/iterator/iterator_facade.hpp>

#include "../exceptions.h"

namespace plask {

/**
 * Sorted, compressed, indexed set of numbers.
 *
 * Set is stored as a sorted vector of segments, each represents a sequence of successive numbers.
 */
template <typename number_t = std::size_t>
struct CompressedSetOfNumbers {

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

    /**
     * Facade which help to develop iterators over CompressedSetOfNumbers.
     *
     * Finall iterator (Derived) can iterate over numbers in set or other classes, and should (directly or indirectly) hold reference to the set.
     *
     * Derived must have set() method which returns <code>const CompressedSetOfNumbers<number_t>&</code>.
     * It may also have dereference() method which returnce @c Reference.
     */
    template <typename Derived, class Value = number_t, class Reference = Value>
    struct ConstIteratorFacade: public boost::iterator_facade<Derived, Value, boost::random_access_traversal_tag, Reference> {

        typedef typename std::vector<Segment>::const_iterator ConstSegmentIterator;

        /// Current segment (which includes current index). It is stored in order to speed up dereference operation.
        ConstSegmentIterator segmentIterator;

        /// Current index.
        std::size_t index;

        /// Construct uninitialized iterator. Don't use it before initialization (which can be done by calling of setIndex method).
        ConstIteratorFacade() {}

        ConstIteratorFacade(std::size_t index, ConstSegmentIterator segmentIterator): segmentIterator(segmentIterator), index(index) {}

        ConstIteratorFacade(std::size_t index) { setIndex(index); }

        /**
         * Get current iterator position (index).
         * @return current iterator position (index)
         */
        std::size_t getIndex() const { return index; }

        void setIndex(std::size_t index) {
            this->index = index;
            segmentIterator = std::upper_bound(_set().segments.begin(), _set().segments.end(), index, Segment::compareByIndexEnd);
        }

        number_t getNumber() const {
            return segmentIterator->numberEnd - segmentIterator->indexEnd + index;
        }

        private: //--- methods used by boost::iterator_facade: ---

        friend class boost::iterator_core_access;

        const CompressedSetOfNumbers<number_t>& _set() const {
            static_cast<const Derived*>(this)->set();
        }

        template <typename OtherT>
        bool equal(const OtherT& other) const {
            return index == other.index;
        }

        void increment() {
            ++index;
            if (index == segmentIterator->indexEnd) ++segmentIterator;
        }

        void decrement() {
            --index;
            if (index < _set().firstIndex(segmentIterator)) --segmentIterator;
        }

        void advance(std::ptrdiff_t to_add) {
            setIndex(index + to_add);
        }

        template <typename OtherT>
        std::ptrdiff_t distance_to(OtherT z) const { return std::ptrdiff_t(z.index) - std::ptrdiff_t(index); }

        number_t dereference() const {  // can be overrwritten by Derived class
            return getNumber();
        }

    };

    class const_iterator: public ConstIteratorFacade<const_iterator> {

        const CompressedSetOfNumbers* _set;

    public:

        template <typename... CtorArgs>
        explicit const_iterator(const CompressedSetOfNumbers& set, CtorArgs&&... ctorArgs)
            : ConstIteratorFacade<const_iterator>(std::forward<CtorArgs>(ctorArgs)...), _set(&set) {}

        const CompressedSetOfNumbers<number_t>& set() const { return *_set; }

    };

    typedef const_iterator iterator;   // we don't support non-const iterators

    const_iterator begin() const { return const_iterator(*this, 0, segments.begin()); }
    const_iterator end() const { return const_iterator(*this, size(), segments.end()); }



    /**
     * Get number of items (numbers) included in the set.
     *
     * Time complexity: constant.
     * @return number of numbers included in the set
     */
    std::size_t size() const { return segments.empty() ? 0 : segments.back().indexEnd; }

    /**
     * Requests the vector of segments to reduce its capacity to fit its size.
     *
     * The request is non-binding, and the container implementation is free to optimize otherwise and leave the vector with a capacity greater than its size.
     */
    void shrink_to_fit() { segments.shrink_to_fit(); }

    /**
     * Requests that the vector of segments capacity be at least enough to contain @p n elements.
     *
     * If @p n is greater than the current vector capacity, the function causes the container to reallocate its storage increasing its capacity to @p n (or greater).
     * In all other cases, the function call does not cause a reallocation and the vector capacity is not affected.
     *
     * @param n minimum capacity for the vector of segments
     */
    void reserve(std::size_t n) { segments.reserve(n); }

    /// Removes all numbers from the set, leaving it empty.
    void clear() { segments.clear(); }

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
     * Check if given @p number is included in the set.
     * @param number item to find
     * @return @c true only if the set includes the @p number
     */
    bool includes(number_t number) const {
        return indexOf(number) != NOT_INCLUDED;
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
