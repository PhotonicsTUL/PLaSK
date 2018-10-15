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

        Segment() = default;

        Segment(number_t numberEnd, number_t indexEnd): numberEnd(numberEnd), indexEnd(indexEnd) {}

        bool operator==(const Segment& other) const {
            return numberEnd == other.numberEnd && indexEnd == other.indexEnd;
        }

        bool operator!=(const Segment& other) const {
            return numberEnd != other.numberEnd || indexEnd != other.indexEnd;
        }

    };

    std::vector<Segment> segments;

    CompressedSetOfNumbers() = default;

    CompressedSetOfNumbers(const std::initializer_list<number_t>& sorted_list) {
        for (number_t n: sorted_list) push_back(n);
    }

    /*number_t sizeOfSegment(std::size_t seg_nr) const {
        return (seg_nr == 0) ? segments.front().indexEnd : (segments[seg_nr].indexEnd - segments[seg_nr-1].indexEnd);
    }*/

    number_t sizeOfSegment(typename std::vector<Segment>::const_iterator it) const {
        return (it == segments.begin()) ? it->indexEnd : (it->indexEnd - (it-1)->indexEnd);
    }

    /**
     * Get the first index in the segment pointed by @p it.
     * @param it iterator to segment
     * @return the first index in the segment @c *it
     */
    number_t firstIndex(typename std::vector<Segment>::const_iterator it) const {
        return (it == segments.begin()) ? 0 : (it-1)->indexEnd;
    }

    /**
     * Get the first number in the segment pointed by @p it.
     * @param it iterator to segment
     * @return first number in the segment @c *it
     */
    number_t firstNumber(typename std::vector<Segment>::const_iterator it) const {
        return it->numberEnd - sizeOfSegment(it);
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
            return static_cast<const Derived*>(this)->set();
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
        if (seg_it == segments.end()) throw OutOfBoundsException("CompressedSetOfNumbers::at", "index", index, 0, this->size()-1);
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
     * Quickly append a number to the end of the set.
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
     * Quickly append a segment to the end of the set.
     *
     * Time complexity: amortized constant.
     * @param num_beg, num_end range [num_beg, num_end) to append; must be non-empty; num_beg-1 must be larger than all numbers already included in the set
     */
    void push_back_segment(number_t num_beg, number_t num_end) {
        if (empty())
            segments.emplace_back(num_end, num_end - num_beg);
        else
            segments.emplace_back(num_end, segments.back().indexEnd + num_end - num_beg);
    }

    /**
     * Append range [num_beg, num_end) to the end of the set.
     *
     * Time complexity: amortized constant.
     * @param num_beg, num_end range [num_beg, num_end) to append; num_beg must be larger than all numbers already included in the set
     */
    void push_back_range(number_t num_beg, number_t num_end) {
        if (num_beg >= num_end) return;
        if (empty())
            segments.emplace_back(num_end, num_end - num_beg);
        else if (segments.back().numberEnd == num_beg) {
            segments.back().numberEnd = num_end;
            segments.back().indexEnd += num_end - num_beg;
        } else
            segments.emplace_back(num_end, segments.back().indexEnd + num_end - num_beg);
    }

    /**
     * Assign a range [num_beg, num_end) to *this.
     * @param num_beg, num_end the range to assing
     */
    void assignRange(number_t num_beg, number_t num_end) {
        segments.resize(1);
        segments.front().numberEnd = num_end;
        segments.front().indexEnd = num_end - num_beg;
    }

    /**
     * Assign a range [0, num_end) to *this.
     * @param num_end end of the range to assing
     */
    void assignRange(number_t num_end) {
        segments.resize(1);
        segments.front().numberEnd = num_end;
        segments.front().indexEnd = 0;
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

    friend std::ostream& operator<<(std::ostream& out, const CompressedSetOfNumbers<number_t>& set) {
        out << "{";
        auto it = set.segments.begin();
        if (it != set.segments.end()) {
            out << (it->numberEnd - it->indexEnd);
            if (it->indexEnd > 1) out << ".." << (it->numberEnd-1);
            ++it;
            while (it != set.segments.end()) {
                auto size = it->indexEnd - (it-1)->indexEnd;
                out << ", " << (it->numberEnd - size);
                if (size > 1) out << ".." << (it->numberEnd-1);
                ++it;
            }
        }
        return out << "}";
    }

    bool operator==(const CompressedSetOfNumbers<number_t>& other) const {
        return segments.size() == other.segments.size() && std::equal(segments.begin(), segments.end(), other.segments.begin());
    }

    bool operator!=(const CompressedSetOfNumbers<number_t>& other) const {
        return !(*this == other);
    }

private:

    /**
     * Try append an end frament of @p a_segment to @p result (if it is included in current segment of B) and update both @p a_segment and @p a_first_number.
     * @param result where to append resulted segment
     * @param a_segment segment in the set A; must meet: a_segment->numberEnd <= numberEnd of current segment of B; it is advanced by this method
     * @param a_segment_end end iterator of the set A
     * @param a_first_number first number in the a_segment; it is updated by this method
     * @param b_first_number first number in the current segment of the set B
     * @return true if output a_segment == a_segment_end
     */
    static bool intersectionStep(CompressedSetOfNumbers<number_t>& result, typename std::vector<Segment>::const_iterator& a_segment, typename std::vector<Segment>::const_iterator a_segment_end, number_t& a_first_number, number_t b_first_number) {
        if (b_first_number < a_segment->numberEnd)  // b_first < a_end <= b_end => [b_first, a_end) is common
            result.push_back_segment(b_first_number, a_segment->numberEnd);
        ++a_segment;
        if (a_segment == a_segment_end) return true;
        a_first_number = a_segment->numberEnd - (a_segment->indexEnd - (a_segment-1)->indexEnd);
        return false;
    }

public:
    /**
     * Calculate an intersection of this and the @p other.
     *
     * Time complexity: O(number of segments in this + number of segments in other)
     * @param other set
     * @return intersection of this and the @p other
     */
    CompressedSetOfNumbers<number_t> intersection(const CompressedSetOfNumbers<number_t>& other) const {
        if (this->empty() || other.empty()) return CompressedSetOfNumbers<number_t>();
        CompressedSetOfNumbers<number_t> result;
        result.reserve(this->size() + other.size());    // enought for sure
        auto this_segment = this->segments.begin();
        auto this_first_number = this_segment->numberEnd - this_segment->indexEnd;
        auto other_segment = other.segments.begin();
        auto other_first_number = other_segment->numberEnd - other_segment->indexEnd;
        while (true) {
            if (this_segment->numberEnd < other_segment->numberEnd) {   // a can be included in b
                if (intersectionStep(result, this_segment, this->segments.end(), this_first_number, other_first_number)) break;
            } else {    // b can be included in a
                if (intersectionStep(result, other_segment, other.segments.end(), other_first_number, this_first_number)) break;
            }
        };
        result.shrink_to_fit();
        return result;
    }

    /**
     * Call f(first, last) for each segment [first, last).
     * @param f two-argument functor
     */
    template <typename F>
    void forEachSegment(F f) const {
        for (auto it = this->segments.begin(); it != this->segments.end(); ++it)
            f(firstNumber(it), it->numberEnd);
    }

    /**
     * Calculate a set with numbers of @c this decreased by @p positions_count. Skip numbers which became negative.
     *
     * Time complexity: linear in number of segments.
     * @param positions_count number of positions to shift
     * @return set with numbers of @c this decreased by @p positions_count (numbers which became negative are skiped)
     */
    CompressedSetOfNumbers<number_t> shiftedLeft(number_t positions_count) const {
        auto seg_it = std::upper_bound(segments.begin(), segments.end(), positions_count, Segment::compareByNumberEnd);
        if (seg_it == segments.end()) return CompressedSetOfNumbers<number_t>();
        CompressedSetOfNumbers<number_t> result;
        result.reserve(segments.end() - seg_it);
        auto first = firstNumber(seg_it);
        auto indexShift = (positions_count > first ? positions_count - first : 0) + firstIndex(seg_it);
        do {
            result.segments.emplace_back(seg_it->numberEnd - positions_count, seg_it->indexEnd - indexShift);
        } while (++seg_it != segments.end());
        return result;
    }

    /**
     * Calculate a transformed version of @c this. Call f(first, last) for each successive segment [first, last) of @c this.
     * Function @p f can change its arguments and this changed range is appended to the end of the resulted set.
     *
     * Time complexity: linear in number of segments.
     * @param f functor which take two number_t& as arguments
     * @return the transformed version of @c this
     */
    template <typename F>
    CompressedSetOfNumbers<number_t> transformed(F f) const {
        CompressedSetOfNumbers<number_t> result;
        result.reserve(segments.size());
        for (auto it = this->segments.begin(); it != this->segments.end(); ++it) {
            number_t beg = firstNumber(it);
            number_t end = it->numberEnd;
            f(beg, end);
            result.push_back_range(beg, end);
        }
        result.shrink_to_fit(); // some segments could be merged or removed
        return result;
    }

};

}   // namespace plask

#endif // PLASK__UTILS_NUMBERS_SET_H

