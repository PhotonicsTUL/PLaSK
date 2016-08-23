#ifndef PLASK__SERIALIZE_H
#define PLASK__SERIALIZE_H

#include <type_traits>

#include <cereal/cereal.hpp>
#include <plask/data.h>

namespace cereal
{
    // Portable serialization for std::size_t

    template <class Archive> inline
    void CEREAL_SAVE_FUNCTION_NAME(Archive& ar, std::size_t const& size)
    {
        size_type portable = static_cast<size_type>(size);
        ar(binary_data(&portable, sizeof(size_type)));
    }

    template <class Archive> inline
    void CEREAL_LOAD_FUNCTION_NAME(Archive& ar, std::size_t& size)
    {
        size_type portable;
        ar(binary_data(&portable, sizeof(size_type)));
        size = static_cast<std::size_t>(portable);
    }


    // Serialization for data of arithmetic using binary serialization, if supported

    template <class Archive, class T> inline
    typename std::enable_if<traits::is_output_serializable<BinaryData<T>, Archive>::value && std::is_arithmetic<T>::value, void>::type
    CEREAL_SAVE_FUNCTION_NAME(Archive& ar, plask::DataVector<T> const& data)
    {
        ar(make_size_tag(static_cast<size_type>(data.size()))); // number of elements
        ar(binary_data(data.data(), data.size() * sizeof(T)));
    }

    template <class Archive, class T> inline
    typename std::enable_if<traits::is_input_serializable<BinaryData<T>, Archive>::value && std::is_arithmetic<T>::value, void>::type
    CEREAL_LOAD_FUNCTION_NAME(Archive& ar, plask::DataVector<T>& data)
    {
        size_type size;
        ar(make_size_tag(size));

        data.reset(static_cast<std::size_t>(size));
        ar(binary_data(data.data(), static_cast<std::size_t>(size) * sizeof(T)));
    }


    // Serialization for non-arithmetic data types

    template <class Archive, class T, class A> inline
    typename std::enable_if<!traits::is_output_serializable<BinaryData<T>, Archive>::value || !std::is_arithmetic<T>::value, void>::type
    CEREAL_SAVE_FUNCTION_NAME(Archive& ar, plask::DataVector<T> const& data)
    {
        ar(make_size_tag(static_cast<size_type>(data.size()))); // number of elements
        for(auto&& v: data) ar(v);
    }

    template <class Archive, class T, class A> inline
    typename std::enable_if<!traits::is_input_serializable<BinaryData<T>, Archive>::value || !std::is_arithmetic<T>::value, void>::type
    CEREAL_LOAD_FUNCTION_NAME(Archive& ar, plask::DataVector<T>& data)
    {
        size_type size;
        ar(make_size_tag(size));

        data.reset(static_cast<std::size_t>(size));
        for(auto&& v: data) ar(v);
    }

}; // namespace cereal



namespace plask {



}; // namespace plask

#endif // PLASK__SSERIALIZE_H
