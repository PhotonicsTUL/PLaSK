/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__LAZYDATA_H
#define PLASK__LAZYDATA_H

#include "data.hpp"
#include "utils/openmp.hpp"

#include "vector/tensor2.hpp"
#include "vector/tensor3.hpp"
#include "utils/iterators.hpp"

namespace plask {

/**
 * Base class for implementation used by lazy data vector.
 *
 * Subclasses must provide thread-safty reading.
 * @tparam T type of data served by the data vector
 */
template <typename T>
struct LazyDataImpl {

    typedef T CellType;

    virtual ~LazyDataImpl() {}

    /**
     * Get index-th value from vector.
     * @param index should be a value from 0 to size()-1
     * @return index-th value from vector
     */
    virtual T at(std::size_t index) const = 0;

    /**
     * Get the number of elements in this vector.
     * @return the number of elements in this vector
     */
    virtual std::size_t size() const = 0;

    /**
     * Get all values as non-lazy vector.
     * @return non-lazy representation of this
     */
    virtual DataVector<const T> getAll() const {
        DataVector<T> res(this->size());
        std::exception_ptr error;
        #pragma omp parallel for
        for (openmp_size_t i = 0; i < res.size(); ++i) {
            if (error) continue;
            try {
                res[i] = this->at(i);
            } catch(...) {
                #pragma omp critical
                error = std::current_exception();
            }
        }
        if (error) std::rethrow_exception(error);
        return res;
    }

    virtual DataVector<T> claim() const {
        return this->getAll().claim();
    }
};

/**
 * Lazy data vector of consts.
 */
template <typename T>
struct ConstValueLazyDataImpl: public LazyDataImpl<T> {

    T value_;
    std::size_t size_;

    ConstValueLazyDataImpl(std::size_t size, const T& value): value_(value), size_(size) {}

    T at(std::size_t) const override { return value_; }

    std::size_t size() const override { return size_; }

    DataVector<const T> getAll() const override { return DataVector<const T>(size_, value_); }

};

/**
 * Wrap DataVector and allow to access to it.
 *
 * getAll() does not copy the wrapped vector.
 */
template <typename T>
struct LazyDataFromVectorImpl: public LazyDataImpl<T> {

    DataVector<const T> vec;

    LazyDataFromVectorImpl(DataVector<const T> vec): vec(vec) {}

    T at(std::size_t index) const override { return vec[index]; }

    std::size_t size() const override { return vec.size(); }

    DataVector<const T> getAll() const override { return vec; }

    DataVector<T> claim() const override { return vec.claim(); }
};

/**
 * Call functor to get data.
 */
template <typename T>
struct LazyDataDelegateImpl: public LazyDataImpl<T> {

  protected:
    std::size_t siz;
    std::function<T(std::size_t)> func;

  public:
    LazyDataDelegateImpl(std::size_t size, std::function<T(std::size_t)> func): siz(size), func(std::move(func)) {}

    T at(std::size_t index) const override { return func(index); }

    std::size_t size() const override { return siz; }
};



/*
 * Base class for lazy data (vector) that holds reference to destination mesh (dst_mesh).
 */
/*template <typename T, typename DstMeshType>
struct LazyDataWithMeshImpl: public LazyDataImpl<T> {

    shared_ptr<const DstMeshType> dst_mesh;

    LazyDataWithMeshImpl(shared_ptr<const DstMeshType> dst_mesh): dst_mesh(dst_mesh) {}

    std::size_t size() const override { return dst_mesh->size(); }

};*/

/**
 * Lazy data (vector).
 *
 * It ownership a pointer to implementation of lazy data (vector) which is of the type LazyDataImpl<T>.
 *
 * Reading from LazyData object is thread-safty.
 */
template <typename T>
class LazyData {

    //TODO change back to unique_ptr when move to lambda capture (C++14) will be supported:
    //std::unique_ptr< const LazyDataImpl<T> > impl;
    shared_ptr< const LazyDataImpl<T> > impl;

public:

    typedef T CellType;
    typedef DataVector<T> DataVectorType;
    typedef DataVector<const T> DataVectorOfConstType;

    /**
     * Construct lazy data vector which use given implementation to provide data.
     * @param impl lazy data vector implementation to use (if nullptr, reset should be called before another methods)
     */
    LazyData(const LazyDataImpl<T>* impl = nullptr): impl(impl) {}

    /**
     * Construct lazy data vector which returns the same value for each index.
     * @param size number of values, size of the vector
     * @param value value which will be returned for each index
     */
    LazyData(std::size_t size, T value): impl(new ConstValueLazyDataImpl<T>(size, value)) {}

    /**
     * Construct lazy data vector which wraps DataVector and allows to access to it.
     * @param data_vector vector to wrap
     */
    LazyData(DataVector<const T> data_vector): impl(new LazyDataFromVectorImpl<T>(data_vector)) {}
    LazyData(DataVector<T> data_vector): impl(new LazyDataFromVectorImpl<T>(data_vector)) {}

    /**
     * Construct lazy data from size and functor.
     * \param size data size
     * \param func function returnig dta at point
     */
    LazyData(std::size_t size, std::function<T(std::size_t)> func):
        impl(new LazyDataDelegateImpl<T>(size, std::move(func))) {}

    void reset(const LazyDataImpl<T>* new_impl = nullptr) { impl.reset(new_impl); }

    void reset(std::size_t size, T value) { impl.reset(new ConstValueLazyDataImpl<T>(size, value)); }

    void reset(DataVector<const T> data_vector) { impl.reset(new LazyDataFromVectorImpl<T>(data_vector)); }
    void reset(DataVector<T> data_vector) { impl.reset(new LazyDataFromVectorImpl<T>(data_vector)); }

    void reset(std::size_t size, std::function<T(std::size_t)> func) {
        impl.reset(new LazyDataDelegateImpl<T>(size, std::move(func)));
    }

    /*LazyData(const LazyData&) = default;
    LazyData(LazyData&&) = default;
    LazyData& operator=(const LazyData&) & = default;
    LazyData& operator=(LazyData&&) & = default;*/

    /**
     * Get index-th value from this vector.
     * @param index should be a value from 0 to size()-1
     * @return index-th value from this vector
     */
    T operator[](std::size_t index) const { return impl->at(index); }

    /**
     * Get index-th value from this vector.
     * @param index should be a value from 0 to size()-1
     * @return index-th value from this vector
     */
    T at(std::size_t index) const { return impl->at(index); }

    DataVector<const T> nonLazy() const { return impl->getAll(); }

    operator DataVector<const T> () const { return impl->getAll(); }

    DataVector<T> claim() const {
        return impl->claim();
        //TODO jeśli używany shared_ptr, to co z przypadkiem gdy impl ma więcej niż 1 referencję? wtedy powinno być zrobione impl->getAll()->claim();
        //  (w przypadku trzymania data vectora zostanie on skopiowany)
    }

    /**
     * Get the number of elements in this vector.
     * @return the number of elements in this vector
     */
    std::size_t size() const { return impl->size(); }

    /**
     * Random access iterator type which allow iterate over all points in this lazy data vector, in order appointed by operator[].
     * This iterator type is indexed, which means that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const LazyData, T > const_iterator;
    typedef const_iterator iterator;
    typedef const_iterator Iterator;

    /// @return iterator referring to the first point in @c this
    const_iterator begin() const { return const_iterator(this, 0); }

    /// @return iterator referring to the past-the-end point in @c this
    const_iterator end() const { return const_iterator(this, size()); }

    bool isNotNull() const { return bool(impl); }

    bool isNull() const { return !impl; }
};

template <typename T, typename ScaleT, typename ReturnedType = typename std::remove_cv<decltype(T()*ScaleT())>::type>
struct ScaledLazyDataImpl: public LazyDataImpl<ReturnedType> {

    LazyData<T> data;

    ScaleT scale;

    ScaledLazyDataImpl(LazyData<T> data, const ScaleT& scale)
        : data(std::move(data)), scale(scale) {}

    ReturnedType at(std::size_t index) const override { return data[index] * scale; }

    std::size_t size() const override { return data.size(); }

};

/**
 * Compute factor of @p data and @p a
 * @param data vector to multiply
 * @param a multiply factor
 * @return \p vec * \p a
 */
template<typename T, typename S>
LazyData<typename ScaledLazyDataImpl<T, S>::CellType> operator*(LazyData<T> data, const S& scale) {
    return new ScaledLazyDataImpl<T, S>(std::move(data), scale);
}

/**
 * Compute factor of @p vec and @p a
 * @param vec vector to multiply
 * @param a multiply factor
 * @return \p vec * \p a
 */
template<typename T, typename S>
LazyData<typename ScaledLazyDataImpl<T, S>::CellType> operator*(const S& scale, LazyData<T> data) {
    return new ScaledLazyDataImpl<T, S>(std::move(data), scale);
}

/**
 * Check if two lazy data vectors are equal.
 * @param a, b vectors to compare
 * @return @c true only if a is equal to b (a[0]==b[0], a[1]==b[1], ...)
 */
template<class T1, class T2> inline
bool operator== ( LazyData<T1> const& a, LazyData<T2> const& b)
{ return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

/**
 * Check if two vectors are equal.
 * @param a, b vectors to compare
 * @return @c true only if a is equal to b (a[0]==b[0], a[1]==b[1], ...)
 */
template<class T1, class T2> inline
bool operator==(LazyData<T1> const& a, DataVector<T2> const& b)
{ return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

/**
 * Check if two vectors are equal.
 * @param a, b vectors to compare
 * @return @c true only if a is equal to b (a[0]==b[0], a[1]==b[1], ...)
 */
template<class T1, class T2> inline
bool operator== (DataVector<T1> const& a, LazyData<T2> const& b) { return b == a; }

/**
 * Check if two data vectors are not equal.
 * @param a, b vectors to compare
 * @return @c true only if a is not equal to b
 */
template<class T1, class T2> inline bool operator!=(LazyData<T1> const& a, LazyData<T2> const& b) { return !(a==b); }
template<class T1, class T2> inline bool operator!=(LazyData<T1> const& a, DataVector<T2> const& b) { return !(a==b); }
template<class T1, class T2> inline bool operator!=(DataVector<T1> const& a, LazyData<T2> const& b) { return !(a==b); }

/**
 * A lexical comparison of two (lazy) data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p a is smaller than the @p b
 */
template<class T1, class T2> inline bool
operator< (LazyData<T1> const& a, LazyData<T2> const& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); }

template<class T1, class T2> inline bool
operator< (DataVector<T1> const& a, LazyData<T2> const& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); }

template<class T1, class T2> inline bool
operator< (LazyData<T1> const& a, DataVector<T2> const& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); }

/**
 * A lexical comparison of two (lazy) data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p b is smaller than the @p a
 */
template<class T1, class T2> inline bool operator> (LazyData<T1> const& a, LazyData<T2> const& b) { return b < a; }
template<class T1, class T2> inline bool operator> (DataVector<T1> const& a, LazyData<T2> const& b) { return b < a; }
template<class T1, class T2> inline bool operator> (LazyData<T1> const& a, DataVector<T2> const& b) { return b < a; }

/**
 * A lexical comparison of two (lazy) data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p a is smaller or equal to @p b
 */
template<class T1, class T2> inline bool operator<= (LazyData<T1> const& a, LazyData<T2> const& b) { return !(b < a); }
template<class T1, class T2> inline bool operator<= (DataVector<T1> const& a, LazyData<T2> const& b) { return !(b < a); }
template<class T1, class T2> inline bool operator<= (LazyData<T1> const& a, DataVector<T2> const& b) { return !(b < a); }

/**
 * A lexical comparison of two (lazy) data vectors.
 * @param a, b vectors to compare
 * @return @c true only if @p b is smaller or equal to @p a
 */
template<class T1, class T2> inline bool operator>= (LazyData<T1> const& a, LazyData<T2> const& b) { return !(a < b); }
template<class T1, class T2> inline bool operator>= (DataVector<T1> const& a, LazyData<T2> const& b) { return !(a < b); }
template<class T1, class T2> inline bool operator>= (LazyData<T1> const& a, DataVector<T2> const& b) { return !(a < b); }

/**
 * Print lazy data vector to stream.
 * @param out output, destination stream
 * @param to_print lazy data vector to print
 * @return @c out
 */
template<class T>
std::ostream& operator<<(std::ostream& out, LazyData<T> const& to_print) {
    out << '['; return print_seq(out, to_print.begin(), to_print.end()) << ']';
}


#ifdef _MSC_VER // MSVC require this while MingW does not accept

#define PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(...) \
    PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_CLASS(LazyData<__VA_ARGS__>) ; \
    PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_STRUCT(LazyDataImpl<__VA_ARGS__>) ;

PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(Tensor3< complex<double> >)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(Tensor2< double >)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(Vec<3, complex<double>>)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(Vec<3, double>)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(Vec<2, double>)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_FOR_LAZY_DATA(double)

#endif


/**
 * LazyData implementation removing NaN from another data.
 * TODO: sanitize also \c inf
 */
template <typename T>
struct SafeDataImpl: public LazyDataImpl<T> {

    const LazyData<T> src;
    const T safe_value;

    /**
     * Create lazy data filter.
     * \param src source lazy data or vector
     * \param sane_value sane value that is returned instead of NaN
     */
    SafeDataImpl(const LazyData<T>& src, const T safe_value=Zero<T>()): src(src), safe_value(safe_value) {}

    std::size_t size() const override { return src.size(); }

    T at(std::size_t i) const override { return remove_nan(src.at(i), safe_value); }
};

/**
 * Make LazyData removing NaN from another lazy data.
 * \param src source lazy data or vector
 * \param sane_value sane value that is returned instead of NaN
 */
template <typename T>
inline LazyData<T> SafeData(const LazyData<T>& src, const T safe_value=Zero<T>()) {
    return LazyData<T>(new SafeDataImpl<T>(src, safe_value));
}

}   // namespace plask

#endif // PLASK__LAZYDATA_H
