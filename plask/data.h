#ifndef PLASK__DATA_H
#define PLASK__DATA_H

/** @file
This file contains classes which can hold (or points to) datas.
*/

#include <iterator>
#include <algorithm>
#include <iostream>
#include <initializer_list>
#include <atomic>
#include <type_traits>
#include <memory>   //std::unique_ptr
#include <cassert>

#include "memalloc.h"

namespace plask {

namespace detail {

    /// Garbage collector info for DataVector
    struct DataVectorGC {
        // Count is atomic so many threads can increment and decrement it at same time.
        // If it is 0, it means that there has been only one DataVector object, so probably one thread uses it.
        std::atomic<unsigned> count;

        std::function<void(void*)>* deleter;

        explicit DataVectorGC(unsigned initial): count(initial), deleter(nullptr) {}

        explicit DataVectorGC(unsigned initial, const std::function<void(void*)>& deleter):
            count(initial), deleter(new std::function<void(void*)>(deleter)) {}

        explicit DataVectorGC(unsigned initial, std::function<void(void*)>&& deleter):
            count(initial), deleter(new std::function<void(void*)>(std::forward<std::function<void(void*)>>(deleter))) {}

        void free(void* data) {
            if (deleter) (*deleter)(data);
            else aligned_free(data);
        }

        ~DataVectorGC() { delete deleter; }
    };
}

/**
 * Store pointer and size. Is like intelligent pointer for plain data arrays.
 *
 * Can work in two modes:
 * - managed — data will be deleted (by aligned_free) by destructor of last DataVector instance which referee to this data (reference counting is using);
 * - non-managed — data will be not deleted by DataVector (so DataVector just refers to external data).
 *
 * In both cases, assign operation and copy constructor of DataVector do not copy the data, but just create DataVectors which refers to the same data.
 * So both these operations are very fast.
 */
template <typename T>
struct DataVector {

    typedef typename std::remove_const<T>::type VT;
    typedef const T CT;

    typedef detail::DataVectorGC Gc;
  private:

    std::size_t size_;                  ///< size of the stored data
    Gc* gc_;                            ///< the reference count for the garbage collector and optional destructor
    T* data_;                           ///< The data of the matrix

    /// Decrease GC counter and free memory if necessary.
    void dec_ref() {    // see http://www.boost.org/doc/libs/1_53_0/doc/html/atomic/usage_examples.html "Reference counting" for optimal memory access description
        if (gc_ && gc_->count.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            gc_->free(reinterpret_cast<void*>(const_cast<VT*>(data_)));
            delete gc_;
        }
    }

    /// Increase GC counter.
    void inc_ref() {
        if (gc_) gc_->count.fetch_add(1, std::memory_order_relaxed);
    }

    friend struct DataVector<VT>;
    friend struct DataVector<CT>;

  public:

    typedef T value_type;               ///< type of the stored data

    typedef T* iterator;                ///< iterator type for the array
    typedef const T* const_iterator;    ///< constant iterator type for the array

    /// Create empty.
    DataVector(): size_(0), gc_(nullptr), data_(nullptr) {}

    /**
     * Create vector of given @p size with uninitialized data values.
     *
     * Reserve memory using aligned_malloc<T>(size) call.
     * @param size total size of the data
     */
    explicit DataVector(std::size_t size): size_(size), gc_(new Gc(1)), data_(aligned_malloc<T>(size)) {}

    /**
     * Create data vector with given @p size and fill all its' cells with given @p value.
     * @param size size of vector
     * @param value initial value for each cell
     */
    DataVector(std::size_t size, const T& value): size_(size) {
        std::unique_ptr<typename std::remove_const<T>::type[], aligned_deleter<T>>
            data_non_const(aligned_malloc<VT>(size));
        std::fill_n(data_non_const.get(), size, value);   // this may throw, but no memory leak than
        gc_ = new Gc(1);
        data_ = data_non_const.release();
    }

    /**
     * Copy constructor. Only makes a shallow copy (doesn't copy data).
     * @param src data source
     */
    DataVector(const DataVector<T>& src): size_(src.size_), gc_(src.gc_), data_(src.data_) { inc_ref(); }

    /**
     * Copy constructor. Only makes a shallow copy (doesn't copy data).
     * @param src data source
     */
    template <typename TS>
    DataVector(const DataVector<TS>& src):
        size_(src.size_), gc_(src.gc_), data_(src.data_) { inc_ref(); }

    /**
     * Assign operator. Only makes a shallow copy (doesn't copy data).
     * @param M data source
     * @return *this
     */
    DataVector<T>& operator=(const DataVector<T>& M) {
        if (this == &M) return *this;   // self protection
        this->dec_ref();                // release old content, this can delete old data
        size_ = M.size_;
        data_ = M.data_;
        gc_ = M.gc_;
        inc_ref();
        return *this;
    }

    /**
     * Assign operator. Only makes a shallow copy (doesn't copy data).
     * @param M data source
     * @return *this
     */
    template <typename TS>
    DataVector<T>& operator=(const DataVector<TS>& M) {
        this->dec_ref();    //release old content, this can delete old data
        size_ = M.size_;
        data_ = M.data_;
        gc_ = M.gc_;
        inc_ref();
        return *this;
    }

    /**
     * Move constructor.
     * @param src data to move
     */
    DataVector(DataVector<T>&& src): size_(src.size_), gc_(src.gc_), data_(src.data_) {
        src.gc_ = nullptr;
    }

    /**
     * Move constructor.
     * @param src data to move
     */
    template <typename TS>
    DataVector(DataVector<TS>&& src): size_(src.size_), gc_(src.gc_), data_(src.data_) {
        src.gc_ = nullptr;
    }

    /**
     * Move operator.
     * @param src data source
     * @return *this
     */
    DataVector<T>& operator=(DataVector<T>&& src) {
        swap(src);
        return *this;
    }

    /**
     * Move operator.
     * @param src data source
     * @return *this
     */
    template <typename TS>
    DataVector<T>& operator=(DataVector<TS>&& src) {
        this->dec_ref();
        size_ = std::move(src.size_);
        gc_ = std::move(src.gc_);
        data_ = std::move(src.data_);
        src.gc_ = nullptr;
        return *this;
    }

    /**
     * Create vector out of existing data.
     * @param size  total size of the existing data
     * @param existing_data pointer to existing data
     */
    template <typename TS>
    DataVector(TS* existing_data, std::size_t size):
        size_(size), gc_(nullptr), data_(existing_data) {}

    /**
     * Create vector out of existing data with guardian.
     * \param size  total size of the existing data
     * \param existing_data pointer to existing data
     * \param deleter function deleting data
     */
    template <typename TS>
    DataVector(TS* existing_data, std::size_t size, const std::function<void(void*)>& deleter):
        size_(size), gc_(new Gc(1, deleter)), data_(existing_data) {
    }

    /**
     * Create vector out of existing data with guardian.
     * \param size  total size of the existing data
     * \param existing_data pointer to existing data
     * \param deleter function deleting data
     */
    template <typename TS>
    DataVector(TS* existing_data, std::size_t size, std::function<void(void*)>&& deleter):
        size_(size), gc_(new Gc(1, std::forward<std::function<void(void*)>>(deleter))), data_(existing_data) {
    }

    /**
     * Create data vector and fill it with data from initializer list.
     * @param init initializer list with data
     */
    DataVector(std::initializer_list<T> init): size_(init.size()), gc_(new Gc(1)), data_(aligned_malloc<T>(size_)) {
        std::copy(init.begin(), init.end(), data_);
    }

    /**
     * Create data vector and fill it with data from initializer list.
     * @param init initializer list with data
     */
    template <typename TS>
    DataVector(std::initializer_list<TS> init): size_(init.size()), gc_(new Gc(1)), data_(aligned_malloc<T>(size_)) {
        std::copy(init.begin(), init.end(), data_);
    }

    /// Delete data if this was last reference to it.
    ~DataVector() { dec_ref(); }

    /**
     * Make this data vector points to nullptr data with 0-size.
     *
     * Same as: DataVector().swap(*this);
     */
    void reset() {
        dec_ref();
        size_ = 0;
        gc_ = nullptr;
        data_ = nullptr;
    }

    /**
     * Change data of this data vector. Same as: DataVector(existing_data, size).swap(*this);
     * @param size  total size of the existing data
     * @param existing_data pointer to existing data
     * @param manage indicates whether the data vector should manage the data and garbage-collect it (with delete[] operator)
     */
    template <typename TS>
    void reset(TS* existing_data, std::size_t size) {
        dec_ref();
        gc_ = nullptr;
        size_ = size;
        data_ = existing_data;
    }

    /**
     * Change data of this data vector. Same as: DataVector(existing_data, size, deleter).swap(*this);
     * \param size  total size of the existing data
     * \param existing_data pointer to existing data
     * \param guardian pointer to the data guardian object created on the heap (it will be deleted automatically)
     */
    template <typename TS>
    void reset(TS* existing_data, std::size_t size, const std::function<void(void*)>& deleter) {
        dec_ref();
        gc_ = Gc(1, deleter);
        size_ = size;
        data_ = existing_data;
    }

    /**
     * Change data of this data vector. Same as: DataVector(existing_data, size, deleter).swap(*this);
     * \param size  total size of the existing data
     * \param existing_data pointer to existing data
     * \param guardian pointer to the data guardian object created on the heap (it will be deleted automatically)
     */
    template <typename TS>
    void reset(TS* existing_data, std::size_t size, std::function<void(void*)>&& deleter) {
        dec_ref();
        gc_ = Gc(1, std::forward<std::function<void(void*)>>(deleter));
        size_ = size;
        data_ = existing_data;
    }

    /**
     * Change data of this data vector to uninitialized data with given @p size.
     *
     * Reserve memory using aligned_malloc<T>(size) call.
     *
     * Same as: DataVector(size).swap(*this);
     * @param size total size of the data
     */
    void reset(std::size_t size) {
        dec_ref();
        data_ = aligned_malloc<T>(size);
        gc_ = new Gc(1);
        size_ = size;
    }

    /**
     * Change data of this to array of given @p size and fill all its' cells with given @p value.
     *
     * Same as: DataVector(size, value).swap(*this);
     * @param size size of vector
     * @param value initial value for each cell
     */
    void reset(std::size_t size, const T& value) {
        std::unique_ptr<VT[], aligned_deleter<T>>
            data_non_const(aligned_malloc<VT>(size));
        std::fill_n(data_non_const.get(), size, value);   // this may throw, than our data will not change
        dec_ref();
        gc_ = new Gc(1);    //this also may throw
        data_ = data_non_const.release();
        size_ = size;
    }

    /**
     * Change data of this to copy of range [begin, end).
     *
     * Same as: DataVector(begin, end).swap(*this);
     * @param begin, end range of data to copy
     */
    template <typename InIterT>
    void reset(InIterT begin, InIterT end) {
        std::unique_ptr<VT[], aligned_deleter<T>>
            data_non_const(aligned_malloc<VT>(size));
        std::copy(begin, end, data_non_const.get());    //this may throw, and than our vector will not changed
        dec_ref();
        gc_ = new Gc(1);
        size_ = std::distance(begin, end);
        data_ = data_non_const.release();
    }

    /**
     * Get iterator referring to the first object in data vector.
     * @return const iterator referring to the first object in data vector
     */
    const_iterator begin() const { return data_; }

    /**
     * Get iterator referring to the first object in data vector.
     * @return iterator referring to the first object in data vector
     */
    iterator begin() { return data_; }

    /**
     * Get iterator referring to the past-the-end object in data vector
     * @return const iterator referring to the past-the-end object in data vector
     */
    const_iterator end() const { return data_ + size_; }

    /**
     * Get iterator referring to the past-the-end object in data vector
     * @return iterator referring to the past-the-end object in data vector
     */
    iterator end() { return data_ + size_; }

    /// @return total size of the matrix/vector
    std::size_t size() const { return size_; }

    /// @return constant pointer to data
    const T* data() const { return data_; }

    /// @return pointer to data
    T* data() { return data_; }

    /**
     * Return reference to the (constant) n-th object of the data.
     * @param n number of object to return
     * @return reference to n-th object of the data
     */
    const T& operator[](std::size_t n) const { assert(n < size_); return data_[n]; }

    /**
     * Return reference to the n-th object of the data.
     * @param n number of object to return
     * @return reference to n-th object of the data
     */
    T& operator[](std::size_t n) { assert(n < size_); return data_[n]; }

    /// \return \c true if vector has any data
    operator bool() const { return data_ != nullptr; };

    /**
     * Make a deep copy of the data.
     * @return new object with manage copy of this data
     */
    DataVector<VT> copy() const {
        DataVector<VT> new_data(size_);
        std::copy(begin(), end(), new_data.begin());
        return new_data;
    }

    /**
     * Check if this is the only one owner of data.
     * @return @c true only if for sure this is the only one owner of data
     */
    bool unique() const {
        return (gc_ != nullptr) && (gc_->count == 1);
    }

    /**
     * Allow to remove const qualifer from data, must be
     * @return non-const version of this which refere to the same data
     */
    DataVector<VT> remove_const() const {
        DataVector<VT> result(const_cast<VT*>(this->data()), this->size());
        result.gc_ = this->gc_;
        result.inc_ref();
        return result;
    }

    /**
     * Make copy of data only if this is not the only one owner of it.
     * @return copy of this: shallow if unique() is @c true, deep if unique() is @c false
     */
    DataVector<VT> claim() const {
        return (unique() && !gc_->deleter)? remove_const() : copy();
    }

    /**
     * Swap all internals of this and @p other.
     * @param other data vector to swap with
     */
    void swap(DataVector<T>& other) {
        std::swap(size_, other.size_);
        std::swap(gc_, other.gc_);
        std::swap(data_, other.data_);
    }

    /**
     * Fill all data using given value.
     * @param value value to fill
     */
    template <typename O>
    void fill(const O& value) {
        std::fill_n(data_, size_, value);
    }

    /**
     * Get sub-array of this which refers to data of this.
     * @param begin_index index of this from which data of sub-array should begin
     * @param subarray_size size of sub-array
     * @return data vector which refers to same data as this and is valid not longer than the data of this
     */
    inline DataVector<T> getSubarrayRef(std::size_t begin_index, std::size_t subarray_size) {
        assert(begin_index + subarray_size <= size_);
        return DataVector<T>(data_ + begin_index, subarray_size);
    }

    /**
     * Get sub-array of this which refers to data of this.
     * @param begin_index index of this from which data of sub-array should begin
     * @param subarray_size size of sub-array
     * @return data vector which refers to the same data as this and is valid not longer than the data of this
     */
    inline DataVector<const T> getSubarrayRef(std::size_t begin_index, std::size_t subarray_size) const {
        assert(begin_index + subarray_size <= size_);
        return DataVector<const T>(data_ + begin_index, subarray_size);
    }

    /**
     * Get sub-array of this which copies the data (fragment) of this.
     * @param begin_index index of this from which data of sub-array should begin
     * @param subarray_size size of sub-array
     * @return data vector which stores new data which are copy of data from this
     */
    inline DataVector<T> getSubarrayCopy(std::size_t begin_index, std::size_t subarray_size) const {
        return getSubarrayRef(begin_index, subarray_size).copy();
    }

    /**
     * Get sub-array of this which refers to data of this.
     * @param begin, end range [begin, end), sub-range of [begin(), end())
     * @return data vector which refers to same data as this and is valid not longer than the data of this
     */
    inline DataVector<T> getSubrangeRef(iterator begin, iterator end) {
        assert(begin() <= begin && begin <= end && end <= end());
        return DataVector<T>(begin, end - begin);
    }

    /**
     * Get sub-array of this which refers to data of this.
     * @param begin, end range [begin, end), sub-range of [begin(), end())
     * @return data vector which refers to same data as this and is valid not longer than the data of this
     */
    inline DataVector<const T> getSubrangeRef(const_iterator begin, const_iterator end) const {
        assert(begin() <= begin && begin <= end && end <= end());
        return DataVector<T>(begin, end-begin);
    }

    /**
     * Get sub-array of this which copies the data (fragment) of this.
     * @param begin, end range [begin, end), sub-range of [begin(), end())
     * @return data vector which stores new data which are copy of the data from this
     */
    inline DataVector<T> getSubrangeCopy(const_iterator begin, const_iterator end) {
        assert(begin() <= begin && begin <= end && end <= end());
        return getSubrangeRef(begin, end).copy();
    }
};

/** \relates DataVector
 * Check if two data vectors are equal.
 * @param a, b vectors to compare
 * @return @c true only if a is equal to b (a[0]==b[0], a[1]==b[1], ...)
 */
template<class T1, class T2> inline
bool operator == ( DataVector<T1> const& a, DataVector<T2> const& b)
{ return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

/** \relates DataVector
 * Check if two data vectors are not equal.
 * @param a, b vectors to compare
 * @return @c true only if a is not equal to b
 */
template<class T1, class T2> inline
bool operator != ( DataVector<T1> const& a, DataVector<T2> const& b) { return !(a==b); }

/** \relates DataVector
 * A lexical comparison of two data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p a is smaller than the @p b
 */
template<class T1, class T2> inline
bool operator< ( DataVector<T1> const& a, DataVector<T2> const& b)
{ return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); }

/** \relates DataVector
 * A lexical comparison of two data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p b is smaller than the @p a
 */
template<class T1, class T2> inline
bool operator> ( DataVector<T1> const& a, DataVector<T2> const& b) { return b < a; }

/** \relates DataVector
 * A lexical comparison of two data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p a is smaller or equal to @p b
 */
template<class T1, class T2> inline
bool operator<= ( DataVector<T1> const& a, DataVector<T2> const& b) { return !(b < a); }

/** \relates DataVector
 * A lexical comparison of two data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p b is smaller or equal to @p a
 */
template<class T1, class T2> inline
bool operator>= ( DataVector<T1> const& a, DataVector<T2> const& b) { return !(a < b); }

/** \relates DataVector
 * Print data vector to stream.
 */
template<class T>
std::ostream& operator<<(std::ostream& out, DataVector<T> const& to_print) {
    out << '[';
    return printRange(out, to_print.begin(), to_print.end()) << ']';
}

/** \relates DataVector
 * Calculate: to_inc[i] += inc_val[i] for each i = 0, ..., min(to_inc.size(), inc_val.size()).
 * @param to_inc vector to increase
 * @param inc_val
 * @return @c *this
 */
template<class T>
DataVector<T>& operator+=(DataVector<T>& to_inc, DataVector<const T> inc_val) {
    std::size_t min_size = std::min(to_inc.size(), inc_val.size());
    for (std::size_t i = 0; i < min_size; ++i)
        to_inc[i] += inc_val[i];
    return to_inc;
}

template <class T>
typename std::remove_cv<T>::type accumulate(const DataVector<const T>& to_accum, typename std::remove_cv<T>::type initial=typename std::remove_cv<T>::type()) {
    for (std::size_t i = 0; i < to_accum.size(); ++i)
        initial += to_accum[i];
    return initial;
}

/**
 * Compute data arithmetic mean
 */
template <class T>
T average(const DataVector<T>& v) {
    return accumulate(v) / v.size();
}

/** \relates DataVector
 * Cast DataVector<const T> into DataVector<T>
 * \param src vector of type DataVector<const T> or DataVector<T>
 * \return data vector of type DataVector<RT>
 * \tparam RT must be equal to T without const
 */
template<typename RT, typename T>
inline DataVector<RT> const_data_cast(const DataVector<T>& src) {
    return src.remove_const();
}

}   // namespace plask

namespace std {
    template <typename T>
    void swap(plask::DataVector<T>& s1, plask::DataVector<T>& s2) { // throw ()
      s1.swap(s2);
    }
}   // namespace std


#endif // PLASK__DATA_H

