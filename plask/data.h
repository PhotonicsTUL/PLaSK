#ifndef PLASK__DATA_H
#define PLASK__DATA_H

/** @file
This file includes classes which can hold (or points to) datas.
*/

#include <iterator>
#include <algorithm>
#include <initializer_list>

namespace plask {

/**
 * Store pointer and size. Is like inteligent pointer for plain data arrays.
 *
 * Can work in two modes:
 * - manage - data will be deleted (by delete[]) by destructor of last DataVector instance which referee to this data (reference counting is using);
 * - non-manage - data will be not deleted by DataVector (so DataVector just referee to external data).
 *
 * In both cases, asign operation and copy constructor of DataVector do not copy the data, but just create DataVectors which referee to the same data.
 * So both of this operations are very fast.
 */
template <typename T>
class DataVector {

    std::size_t size_;                  ///< size of the stored data
    unsigned* gc_;                      ///< the reference count for the garbage collector
    T* data_;                           ///< The data of the matrix

    /// Decrease GC counter and free memory if necessary.
    void dec_ref() {
        if (gc_ && --(*gc_) == 0) {
            delete gc_;
            delete[] data_;
        }
    }

    /// Increace GC counter.
    void inc_ref() {
        if (gc_) ++(*gc_);
    }

  public:

    typedef T type;                     ///< type of the stored data

    typedef T* iterator;                ///< iterator type for the array
    typedef const T* const_iterator;    ///< constant iterator type for the array

    /// Create empty.
    DataVector() : size_(0), gc_(nullptr), data_(nullptr) {}

    /**
     * Create vector of given @p size.
     *
     * Reserve memory using new T[size] call.
     * @param size total size of the data
     */
    DataVector(std::size_t size) : size_(size), gc_(new unsigned(1)), data_(new T[size]) {}

    /**
     * Create data vector with given @p size and fill all its' cells with given @p value.
     * @param size size of vector
     * @param value initial value for each cell
     */
    DataVector(std::size_t size, const T& value): size_(size), gc_(new unsigned(1)), data_(new T[size]) {
        std::fill(begin(), end(), value);
    }

    /**
     * Copy constructor. Only makes shallow copy (doesn't copy data).
     * @param src data source
     */
    DataVector(const DataVector& src): size_(src.size_), gc_(src.gc_), data_(src.data_) { inc_ref(); }

    /**
     * Assign operator. Only makes shallow copy (doesn't copy data).
     * @param M data source
     * @return *this
     */
    DataVector<T>& operator=(const DataVector<T>& M) {
        if (this == &M) return;
        this->dec_ref();
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
    DataVector(DataVector&& src): size_(src.size_), gc_(src.gc_), data_(src.data_) {
        src.gc_ = nullptr;
    }

    /**
     * Move operator.
     * @param src data source
     * @return *this
     */
    DataVector<T>& operator=(DataVector&& src) {
        /*this->dec_ref();
        size_ = src.size_;
        data_ = src.data_;
        gc_ = src.gc_;
        src.gc_ = nullptr;*/
        swap(src);
        return *this;
    }

    /**
     * Create vector out of existing data.
     * @param size  total size of the existing data
     * @param existing_data pointer to existing data
     * @param manage indicates whether the data vector should manage the data and garbage-collect it (with delete[] operator)
     */
    DataVector(T* existing_data, std::size_t size, bool manage = false)
        : size_(size), gc_(manage ? new unsigned(1) : nullptr), data_(existing_data) {}

    /**
     * Create data vector and fill it with data from initializer list.
     * @param init initializer list with data
     */
    DataVector(std::initializer_list<T> init): size_(init.size()), gc_(new unsigned(1)), data_(new T[size_]) {
        std::copy(init.begin(), init.end(), data_);
    }

    /**
     * Create vector which data are copy of range [begin, end).
     * @param begin, end range of data to copy
     */
    template <typename InIterT>
    DataVector(InIterT begin, InIterT end): size_(std::distance(begin, end)), gc_(new unsigned(1)), data_(new T[size_]) {
        std::copy(begin, end, data_);
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
        gc_ = 0;
        data_ = nullptr;
    }

    /**
     * Chenge data of this data vector. Same as: DataVector(existing_data, size, manage).swap(*this);
     * @param size  total size of the existing data
     * @param existing_data pointer to existing data
     * @param manage indicates whether the data vector should manage the data and garbage-collect it (with delete[] operator)
     */
    void reset(T* existing_data, std::size_t size, bool manage = false) {
        dec_ref();
        size_ = size;
        gc_ = manage ? new unsigned(1) : nullptr;
        data_ = existing_data;
    }

    /**
     * Chenge data of this data vector to unitilized data with given @p size.
     *
     * Reserve memory using new T[size] call.
     *
     * Same as: DataVector(size).swap(*this);
     * @param size total size of the data
     */
    void reset(std::size_t size) {
        dec_ref();
        size_ = size;
        gc_ = new unsigned(1);
        data_ = new T[size];
    }

    /**
     * Chenge data of this to array of given @p size and fill all its' cells with given @p value.
     *
     * Same as: DataVector(size, value).swap(*this);
     * @param size size of vector
     * @param value initial value for each cell
     */
    void reset(std::size_t size, const T& value) {
        dec_ref();
        size_ = size;
        gc_ = new unsigned(1);
        data_ = new T[size];
        std::fill(begin(), end(), value);
    }

    /**
     * Chenge data of this to copy of range [begin, end).
     *
     * Same as: DataVector(begin, end).swap(*this);
     * @param begin, end range of data to copy
     */
    template <typename InIterT>
    void reset(InIterT begin, InIterT end) {
        size_ = std::distance(begin, end);
        gc_ = new unsigned(1);
        data_ = new T[size_];
        std::copy(begin, end, data_);
    }

    /**
     * Get iterator referring to the first element in data vector.
     * @return const iterator referring to the first element in data vector
     */
    const_iterator begin() const { return data_; }

    /**
     * Get iterator referring to the first element in data vector.
     * @return iterator referring to the first element in data vector
     */
    iterator begin() { return data_; }

    /**
     * Get iterator referring to the past-the-end element in data vector
     * @return const iterator referring to the past-the-end element in data vector
     */
    const_iterator end() const { return data_ + size_; }

    /**
     * Get iterator referring to the past-the-end element in data vector
     * @return iterator referring to the past-the-end element in data vector
     */
    iterator end() { return data_ + size_; }

    /// @return total size of the matrix/vector
    std::size_t size() const { return size_; }

    /// @return constant pointer to data
    const T* data() const { return data_; }

    /// @return pointer to data
    T* data() { return data_; }

    /**
     * Return n-th element of the data.
     * @param n number of element to return
     */
    const T& operator [](std::size_t n) const { return data_[n]; }

    /**
     * Return reference to the n-th element of the data.
     * @param n number of element to return
     */
    T& operator [](std::size_t n) { return data_[n]; }

    /**
     * Make a deep copy of the data.
     * @return new object with manage copy of this data
     */
    DataVector<T> copy() const {
        T* new_data = new T[size_];
        std::copy(begin(), end(), new_data);
        return DataVector<T>(new_data, size_, true);
    }

    /**
     * Check if this is the only one owner of data.
     * @return @c true only if this is the only one owner of data
     */
    bool unique() const {
        return (gc_ != nullptr) && (*gc_ == 1);
    }

    /**
     * Make copy of data only if this is not the only one owner of it.
     * @return copy of this: shallow if unique() is @c true, deep if unique() is @c false
     */
    DataVector<T> claim() const {
        return unique() ? *this : copy();
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
};

}   // namespace plask

namespace std {
    template <typename T>
    void swap(plask::DataVector<T>& s1, plask::DataVector<T>& s2) { // throw ()
      s1.swap(s2);
    }
}   // namespace std

#endif // PLASK__DATA_H
