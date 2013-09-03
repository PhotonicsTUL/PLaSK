#ifndef PLASK__PROVIDERFOR_H
#define PLASK__PROVIDERFOR_H

/** @file
This file contains classes and templates which allow to generate providers and receivers.
@see @ref providers
*/

#include "provider.h"
#include "../utils/stl.h"   // VariadicTemplateTypesHolder

namespace plask {

/**
 * Type of properties.
 * @see @ref providers
 */
enum PropertyType {
    SINGLE_VALUE_PROPERTY = 0,          ///< Single value property
    MULTI_VALUE_PROPERTY = 1,           ///< Multiple values property
    FIELD_PROPERTY = 2,                 ///< Property for field which can be interpolated
    MULTI_FIELD_PROPERTY = 3            ///< Property for multiple fields which can be interpolated
};

template <PropertyType prop_type>
struct PropertyTypeToProviderName {
    static constexpr const char* value = "undefined";
};

template <>
struct PropertyTypeToProviderName<SINGLE_VALUE_PROPERTY> {
    static constexpr const char* value = "undefined value";
};

template <>
struct PropertyTypeToProviderName<MULTI_VALUE_PROPERTY> {
    static constexpr const char* value = "undefined value";
};

template <>
struct PropertyTypeToProviderName<FIELD_PROPERTY> {
    static constexpr const char* value = "undefined field";
};

template <>
struct PropertyTypeToProviderName<MULTI_FIELD_PROPERTY> {
    static constexpr const char* value = "undefined field";
};

/**
 * Helper class which makes it easier to define property tags class.
 *
 * Property tags class are used for ProviderFor and ReceiverFor templates instantiations.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 *
 * @tparam _propertyType type of property
 * @tparam _ValueType2D type of value (or part of this type) which will be provided in 2D space
 * @tparam _ValueType3D type of value (or part of this type) which will be provided in 3D space
 * @tparam _ExtraParams type of extra parameters passed to provider
 */
template <PropertyType _propertyType, typename _ValueType2D, typename _ValueType3D, typename... _ExtraParams>
struct Property {
    /// Type of property.
    static const PropertyType propertyType = _propertyType;

    /// Type of provided value in 2D space.
    typedef _ValueType2D ValueType2D;

    /// Type of provided value in 3D space.
    typedef _ValueType3D ValueType3D;

    /// Name of the property
    static constexpr const char* NAME = PropertyTypeToProviderName<_propertyType>::value;

    /// Return default value of the property (usually zero) in 2D space
    static inline ValueType2D getDefaultValue2D() { return ValueType2D(); }

    /// Return default value of the property (usually zero) in 3D space
    static inline ValueType3D getDefaultValue3D() { return ValueType3D(); }

    /// Extra parameters passed as arguments to provider to get value
    typedef VariadicTemplateTypesHolder<_ExtraParams...> ExtraParams;

    /// @c true only if property use same value type in 2D and 3D space
    static constexpr bool hasUniqueValueType = false;

};

template <PropertyType _propertyType, typename _ValueType, typename... _ExtraParams>
struct Property<_propertyType, _ValueType, _ValueType, _ExtraParams...> {
    /// Type of property.
    static const PropertyType propertyType = _propertyType;

    /// Type of provided value.
    typedef _ValueType ValueType;

    /// Type of provided value in 2D space.
    typedef _ValueType ValueType2D;

    /// Type of provided value in 3D space.
    typedef _ValueType ValueType3D;

    /// Name of the property
    static constexpr const char* NAME = PropertyTypeToProviderName<_propertyType>::value;

    /// Return default value of the property (usually zero)
    static inline ValueType getDefaultValue() { return ValueType(); }

    /// Extra parameters passed as arguments to provider to get value
    typedef VariadicTemplateTypesHolder<_ExtraParams...> ExtraParams;

    /// @c true only if property use same value type in 2D and 3D space
    static constexpr bool hasUniqueValueType = true;

    /**
     * Convert value in 3D space to 2D space
     * @param v value in 3D space
     * @return @p p converted to 2D space
     */
    static const ValueType2D& value3Dto2D(const ValueType3D& v) { return v; }

    /**
     * Convert value in 2D space to 2D space
     * @param v value in 2D space
     * @return @p p converted to 3D space
     */
    static const ValueType3D& value2Dto3D(const ValueType2D& v) { return v; }
};

template <typename PropertyTag, bool hasUniqueValueType>
struct PropertyVecConverterImpl {};

template <typename PropertyTag>
struct PropertyVecConverterImpl<PropertyTag, true> {
    static const DataVector<const typename PropertyTag::ValueType2D>& from3Dto2D(const DataVector<const typename PropertyTag::ValueType3D>& datavec) {
        return datavec;
    }
    static const DataVector<typename PropertyTag::ValueType2D>& from3Dto2D(const DataVector<typename PropertyTag::ValueType3D>& datavec) {
        return datavec;
    }
    static const DataVector<const typename PropertyTag::ValueType3D>& from2Dto3D(const DataVector<const typename PropertyTag::ValueType2D>& datavec) {
        return datavec;
    }
    static const DataVector<typename PropertyTag::ValueType3D>& from2Dto3D(const DataVector<typename PropertyTag::ValueType2D>& datavec) {
        return datavec;
    }
};


template <typename PropertyTag>
struct PropertyVecConverterImpl<PropertyTag, false> {
    static DataVector<typename PropertyTag::ValueType2D> from3Dto2D(const DataVector<const typename PropertyTag::ValueType3D>& datavec) {
        DataVector<typename PropertyTag::ValueType2D> result(datavec.size());
        for (std::size_t i = 0; i < datavec.size(); ++i)
            result[i] = PropertyTag::value3Dto2D(datavec[i]);
        return result;
    }
    static DataVector<typename PropertyTag::ValueType3D> from2Dto3D(const DataVector<const typename PropertyTag::ValueType2D>& datavec) {
        DataVector<typename PropertyTag::ValueType3D> result(datavec.size());
        for (std::size_t i = 0; i < datavec.size(); ++i)
            result[i] = PropertyTag::value2Dto3D(datavec[i]);
        return result;
    }
};

/**
 * Convert data vector from type of proprty in 3D to 2D space.
 */
template <typename PropertyTag, typename VectorType>
inline auto PropertyVec3Dto2D(const VectorType& datavec) -> decltype(PropertyVecConverterImpl<PropertyTag, PropertyTag::hasUniqueValueType>::from3Dto2D(datavec)) {
    return PropertyVecConverterImpl<PropertyTag, PropertyTag::hasUniqueValueType>::from3Dto2D(datavec);
}

template <typename PropertyTag, typename VectorType>
inline auto PropertyVec2Dto3D(const VectorType& datavec) -> decltype(PropertyVecConverterImpl<PropertyTag, PropertyTag::hasUniqueValueType>::from2Dto3D(datavec)) {
    return PropertyVecConverterImpl<PropertyTag, PropertyTag::hasUniqueValueType>::from2Dto3D(datavec);
}

/// Describe property in given space. Don't use it directly, but use PropertyAt.
template <typename PropertyTag, int DIM, bool hasUniqueValueType>
struct PropertyAtImpl {};

/// Describe property in 2D space. Don't use it directly, but use PropertyAt.
template <typename PropertyTag>
struct PropertyAtImpl<PropertyTag, 2, true> {
    typedef typename PropertyTag::ValueType2D ValueType;

    static ValueType getDefaultValue() { return PropertyTag::getDefaultValue(); }
};

template <typename PropertyTag>
struct PropertyAtImpl<PropertyTag, 2, false> {
    typedef typename PropertyTag::ValueType2D ValueType;

    static ValueType getDefaultValue() { return PropertyTag::getDefaultValue2D(); }
};

/// Describe property in 3D space. Don't use it directly, but use PropertyAt.
template <typename PropertyTag>
struct PropertyAtImpl<PropertyTag, 3, true> {
    typedef typename PropertyTag::ValueType3D ValueType;

    static ValueType getDefaultValue() { return PropertyTag::getDefaultValue(); }
};

template <typename PropertyTag>
struct PropertyAtImpl<PropertyTag, 3, false> {
    typedef typename PropertyTag::ValueType3D ValueType;

    static ValueType getDefaultValue() { return PropertyTag::getDefaultValue3D(); }
};

/**
 * Describe property type in given space.
 *
 * Includes:
 * - ValueType - typedef to value provided by tag in given space.
 * - getDefaultValue() - static method which returns default value in given space.
 */
template <typename PropertyTag, int dim>
using PropertyAt = PropertyAtImpl<PropertyTag, dim, PropertyTag::hasUniqueValueType>;

template <typename PropertyTag, typename Space>
struct PropertyAtSpace: public PropertyAt<PropertyTag, Space::DIM> {};

template <typename PropertyTag>
struct PropertyAtSpace<PropertyTag, void>: public PropertyAt<PropertyTag, 2> {
    static_assert(PropertyTag::hasUniqueValueType, "Space was not given in PropertyAtSpace for property which has different types of values in 2D and 3D.");
};


/**
 * Helper class which makes it easier to define property tags class for single value (double type by default) properties.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct SingleValueProperty: public Property<SINGLE_VALUE_PROPERTY, ValueT, ValueT, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags class for multiple value (double type by default) properties.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct MultiValueProperty: public Property<MULTI_VALUE_PROPERTY, ValueT, ValueT, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags class for fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct FieldProperty: public Property<FIELD_PROPERTY, ValueT, ValueT, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags class for multiple fields.
 *
 * Properties tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must by separate class - always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct MultiFieldProperty: public Property<MULTI_FIELD_PROPERTY, ValueT, ValueT, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags classes for vectorial fields that can be interpolated.
 *
 * Properties defined with this tag has another type of value in 2D and 3D space:
 * - ValueT_2D in 2D space,
 * - ValueT_3D in 3D space.
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
template<typename ValueT_2D, typename ValueT_3D, typename... _ExtraParams>
struct CustomFieldProperty: public Property<FIELD_PROPERTY, ValueT_2D, ValueT_3D, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags classes for multiple vectorial fields that can be interpolated.
 *
 * Properties defined with this tag has another type of value in 2D and 3D space:
 * - ValueT_2D in 2D space,
 * - ValueT_3D in 3D space.
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
template<typename ValueT_2D, typename ValueT_3D, typename... _ExtraParams>
struct MultiCustomFieldProperty: public Property<MULTI_FIELD_PROPERTY, ValueT_2D, ValueT_3D, _ExtraParams...> {};

/**
 * Helper class which makes it easier to define property tags classes for vectorial fields that can be interpolated.
 *
 * Properties defined with this tag has another type of value in 2D and 3D space:
 * - Vec<2, ValueT> in 2D space,
 * - Vec<3, ValueT> in 3D space.
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct VectorFieldProperty: public Property<FIELD_PROPERTY, Vec<2, ValueT>, Vec<3, ValueT>, _ExtraParams...> {

    /**
     * Convert value in 3D space to 2D space by removing component.
     * @param v value in 3D space
     * @return @p p converted to 2D space
     */
    static Vec<2, ValueT> value3Dto2D(const Vec<3, ValueT>& v) { return vec<2>(v); }

    /**
     * Convert value in 2D space to 2D space by adding zeroed component.
     * @param v value in 2D space
     * @return @p p converted to 3D space
     */
    static Vec<3, ValueT> value2Dto3D(const Vec<2, ValueT>& v) { return vec(v, ValueT()); }
};

/**
 * Helper class which makes it easier to define property tags classes for multiple vectorial fields that can be interpolated.
 *
 * Properties defined with this tag has another type of value in 2D and 3D space:
 * - Vec<2, ValueT> in 2D space,
 * - Vec<3, ValueT> in 3D space.
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
template<typename ValueT = double, typename... _ExtraParams>
struct MultiVectorFieldProperty: public Property<MULTI_FIELD_PROPERTY, Vec<2, ValueT>, Vec<3, ValueT>, _ExtraParams...> {

    /**
     * Convert value in 3D space to 2D space by removing component.
     * @param v value in 3D space
     * @return @p p converted to 2D space
     */
    static Vec<2, ValueT> value3Dto2D(const Vec<3, ValueT>& v) { return vec<2>(v); }

    /**
     * Convert value in 2D space to 2D space by adding zeroed component.
     * @param v value in 2D space
     * @return @p p converted to 3D space
     */
    static Vec<3, ValueT> value2Dto3D(const Vec<2, ValueT>& v) { return vec(v, ValueT()); }
};

/**
 * Helper class which makes it easier to define property tags classes for scalar fields (fields of doubles).
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
typedef FieldProperty<double> ScalarFieldProperty;

/**
 * Helper class which makes it easier to define property tags classes for multiple scalar fields (fields of doubles).
 *
 * Property tag class can be subclass of this, but never should be typedefs to this
 * (tag class for each property must be a separate class — always use different types for different properties).
 */
typedef MultiFieldProperty<double> MultiScalarFieldProperty;

/**
 * Specializations of this class are implementations of providers for given property tag class and this tag properties.
 *
 * Don't use this class directly. Use plask::Provider class or plask::ProviderFor template.
 */
template <typename PropertyT, PropertyType propertyType, typename spaceType, typename VariadicTemplateTypesHolder>
struct ProviderImpl {};

/**
 * Specializations of this class define implementations of providers for given property tag:
 * - ProviderFor<PropertyT, SpaceT> is abstract, base class which inherited from Provider;
 * - ProviderFor<PropertyT, SpaceT>::Delegate is class inherited from ProviderFor<PropertyT, SpaceT> which delegates all request to functor given as constructor parameter;
 * - ProviderFor<PropertyT, SpaceT>::WithValue is class inherited from ProviderFor<PropertyT, SpaceT> which stores provided value (has value field) and know if it was initialized;
 * - ProviderFor<PropertyT, SpaceT>::WithDefaultValue is class inherited from ProviderFor<PropertyT, SpaceT> which stores provided value (has value field) and doesn't know if it was initialized (should always have reasonable default value).
 * @tparam PropertyT property tag class (describe physical property)
 * @tparam SpaceT type of space, required (and allowed) only for fields properties
 * @see plask::Temperature (contains example); @ref providers
 */
template <typename PropertyT, typename SpaceT = void>
struct ProviderFor: public ProviderImpl<PropertyT, PropertyT::propertyType, SpaceT, typename PropertyT::ExtraParams> {

    typedef PropertyT PropertyTag;
    typedef SpaceT SpaceType;

    /// Delegate all constructors to parent class.
    template<typename ...Args>
    ProviderFor(Args&&... params)
    : ProviderImpl<PropertyT, PropertyT::propertyType, SpaceT, typename PropertyT::ExtraParams>(std::forward<Args>(params)...) {
    }

};
//TODO redefine ProviderFor using template aliases (require gcc 4.7), and than fix ReceiverFor
//template <typename PropertyT, typename SpaceT = void>
//using ProviderFor = ProviderImpl<PropertyT, PropertyT::propertyType, SpaceT, typename PropertyT::ExtraParams>;




/**
 * Specializations of this class are implementations of Receiver for given property tag.
 * @tparam PropertyT property tag class (describe physical property)
 * @tparam SpaceT type of space, required (and allowed) only for fields properties
 */
template <typename PropertyT, typename SpaceT = void>
struct ReceiverFor: public Receiver<ProviderImpl<PropertyT, PropertyT::propertyType, SpaceT, typename PropertyT::ExtraParams>> {
    ReceiverFor & operator=(const ReceiverFor&) = delete;
    ReceiverFor(const ReceiverFor&) = delete;
    ReceiverFor() = default;

    typedef PropertyT PropertyTag;
    typedef SpaceT SpaceType;
    typedef typename PropertyAtSpace<PropertyT, SpaceT>::ValueType ValueType;

    /**
     * Set provider for this to provider of constant.
     *
     * Use ProviderT::ConstProviderType as provider of const type.
     * @param v value which should be provided for this receiver
     * @return *this
     */
    ReceiverFor<PropertyT, SpaceT>& operator=(const ValueType& v) {
        this->setConstValue(v);
        return *this;
    }

    /**
     * Set provider to internal provider of some value.
     * \param value value to set
     */
    template <typename... Args, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == SINGLE_VALUE_PROPERTY>::type
    setValue(Args&&... value) {
        this->setProvider(new typename ProviderFor<PropertyTag>::WithValue(std::forward<Args>(value)...), true);
    }

    /**
     * Set provider to internal provider of some value.
     * \param value value to set
     */
    template <typename... Args, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == MULTI_VALUE_PROPERTY>::type
    setValue(const ValueType& value) {
        this->setProvider(new typename ProviderFor<PropertyTag>::WithValue(value), true);
    }

    /**
     * Set provider to internal provider of some value.
     * \param value value to set
     */
    template <typename... Args, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == MULTI_VALUE_PROPERTY>::type
    setValues(Args&&... value) {
        this->setProvider(new typename ProviderFor<PropertyTag>::WithValue(std::forward<Args>(value)...), true);
    }

    /**
     * Set provider to internal to provider of given field.
     * \param data data with field values in mesh points
     * \param mesh mesh value
     */
    template <typename MeshPtrT, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == FIELD_PROPERTY || propertyType == MULTI_FIELD_PROPERTY>::type
    setValue(typename ProviderFor<PropertyTag, SpaceType>::ProvidedType data, const MeshPtrT& mesh) {
        if (data.size() != mesh->size())
            throw BadMesh("ReceiverFor::setValues()", "Mesh size (%2%) and data size (%1%) do not match", data.size(), mesh->size());
        this->setProvider(new typename ProviderFor<PropertyTag, SpaceType>::template WithValue<MeshPtrT>(data, mesh), true);
    }

    /**
     * Set provider to internal to provider of given field.
     * \param data data with field values in mesh points
     * \param mesh mesh value
     */
    template <typename MeshPtrT, typename Iterator, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == MULTI_FIELD_PROPERTY>::type
    setValues(Iterator begin, Iterator end, const MeshPtrT& mesh) {
        size_t i = 0;
        for (Iterator it = begin; it != end; ++it, ++i )
            if (*it.size() != mesh->size())
                throw BadMesh("ReceiverFor::setValues()", "Mesh size (%2%) and data[%3%] size (%1%) do not match", it->size(), mesh->size(), i);
        this->setProvider(new typename ProviderFor<PropertyTag, SpaceType>::template WithValue<MeshPtrT>(begin, end, mesh), true);
    }

    /**
     * Set provider to internal to provider of given field.
     * \param data data with field values in mesh points
     * \param mesh mesh value
     */
    template <typename MeshPtrT, PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == MULTI_FIELD_PROPERTY>::type
    setValue(const std::vector<typename ProviderFor<PropertyTag, SpaceType>::ProvidedType>& data, const MeshPtrT& mesh) {
        for (auto it = data.begin(); it != data.end(); ++it)
            if (it->size() != mesh->size())
                throw BadMesh("ReceiverFor::setValues()", "Mesh size (%2%) and data[%3%] size (%1%) do not match", it->size(), mesh->size(), it-data.begin());
        this->setProvider(new typename ProviderFor<PropertyTag, SpaceType>::template WithValue<MeshPtrT>(data.begin(), data.end(), mesh), true);
    }

    /**
     * Return number of elements provided by provider
     */
    template <PropertyType propertyType = PropertyTag::propertyType>
    typename std::enable_if<propertyType == MULTI_VALUE_PROPERTY || propertyType == MULTI_FIELD_PROPERTY, size_t>::type
    size() const {
        this->ensureHasProvider();
        return this->provider->size();
    }
    
    static_assert(!(std::is_same<SpaceT, void>::value && (PropertyT::propertyType == FIELD_PROPERTY || PropertyT::propertyType == MULTI_FIELD_PROPERTY)),
                  "Receivers for fields properties require SpaceT. Use ReceiverFor<propertyTag, SpaceT>, where SpaceT is one of the classes defined in <plask/geometry/space.h>.");
    static_assert(!(!std::is_same<SpaceT, void>::value && (PropertyT::propertyType == SINGLE_VALUE_PROPERTY || PropertyT::propertyType == MULTI_VALUE_PROPERTY)),
                  "Receivers for single value properties doesn't need SpaceT. Use ReceiverFor<propertyTag> (without second template parameter).");

    ///**
    //    * Set provider to of derived type
    //    * \param provider new provider
    //    */
    //template <typename OtherProvidersT>
    //void setProvider(OtherProviderT* provider) {
    //    auto provider = new ProviderFor<OtherProviderT::PropertyTag>::Delegate([&provider](){})
    //}

};
//struct ReceiverFor: public Receiver< ProviderFor<PropertyT> > {};

/**
 * Partial specialization which implements abstract provider class which provides a single value, typically one double.
 *
 * @tparam PropertyT
 * @tparam ValueT type of provided value
 * @tparam SpaceT ignored
 */
template <typename PropertyT, typename SpaceT, typename... _ExtraParams>
struct ProviderImpl<PropertyT, SINGLE_VALUE_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >: public SingleValueProvider<typename PropertyAtSpace<PropertyT, SpaceT>::ValueType, _ExtraParams...> {

    static constexpr const char* NAME = PropertyT::NAME;
    virtual const char* name() const { return NAME; }

    static_assert(std::is_same<SpaceT, void>::value,
                  "Providers for single value properties doesn't need SpaceT. Use ProviderFor<propertyTag> (without second template parameter).");

    typedef typename PropertyAtSpace<PropertyT, SpaceT>::ValueType ValueType;

    /// Type of provided value.
    typedef typename SingleValueProvider<ValueType>::ProvidedType ProvidedType;

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() returns its held value.
     * It always has a value.
     *
     * It ignores eventual extra parameters.
     */
    struct WithDefaultValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value.
        typedef ValueType ProvidedType;

        /// Provided value.
        ProvidedType value;

        /// Delegate all constructors to value.
        template<typename ...Args>
        WithDefaultValue(Args&&... params): value(std::forward<Args>(params)...) {}

        /**
         * Set new value.
         * @param v new value
         * @return *this
         */
        WithDefaultValue& operator=(const ValueType& v) {
            value = v;
            return *this;
        }

        /**
         * Get provided value.
         * @return provided value
         */
        virtual ProvidedType operator()(_ExtraParams...) const { return value; }
    };

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() return its held value.
     *
     * Its value is optional and can throw exception if value was not assigned before request to it.
     *
     * It ignores eventual extra parameters.
     */
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value.
        typedef ValueType ProvidedType;

        /// Provided value.
        boost::optional<ProvidedType> value;

        /// Reset value to be uninitialized.
        void invalidate() { value.reset(); }

        /**
         * Check if this has value / is initialized.
         * @return @c true only if this is initialized (has value)
         */
        bool hasValue() const { return value; }

        /// Throw NoValue exception if value is not initialized.
        void ensureHasValue() const {
            if (!hasValue()) throw NoValue(NAME);
        }

        /// Construct value
        WithValue(const ProvidedType& value): value(value) {}

        /// Construct value
        WithValue(ProvidedType&& value): value(value) {}

        /// Create empty boost::optional value.
        WithValue() {}

        /**
         * Set new value.
         * @param v new value
         * @return *this
         */
        WithValue& operator=(const ValueType& v) {
            value.reset(v);
            return *this;
        }

        /**
         * Get provided value.
         * @return provided value
         * @throw NoValue if value is empty boost::optional
         */
        virtual ProvidedType operator()(_ExtraParams...) const {
            ensureHasValue();
            return *value;
        }
    };

    /**
     * Implementation of one value provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(_ExtraParams...)> Delegate;

    /// Used by receivers as const value provider, see Receiver::setConst
    typedef WithValue ConstProviderType;

};

/**
 * Partial specialization which implements abstract provider class which provides a single value, typically one double.
 *
 * @tparam PropertyT
 * @tparam ValueT type of provided value
 * @tparam SpaceT ignored
 */
template <typename PropertyT, typename SpaceT, typename... _ExtraParams>
struct ProviderImpl<PropertyT, MULTI_VALUE_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >: public MultiValueProvider<typename PropertyAtSpace<PropertyT, SpaceT>::ValueType, _ExtraParams...> {

    static constexpr const char* NAME = PropertyT::NAME;
    virtual const char* name() const { return NAME; }

    static_assert(std::is_same<SpaceT, void>::value,
                  "Providers for single value properties doesn't need SpaceT. Use ProviderFor<propertyTag> (without second template parameter).");

    typedef typename PropertyAtSpace<PropertyT, SpaceT>::ValueType ValueType;

    /// Type of provided value.
    typedef typename MultiValueProvider<ValueType>::ProvidedType ProvidedType;

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() returns its held value.
     * It always has a value.
     *
     * It ignores eventual extra parameters.
     */
    struct WithDefaultValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value
        typedef ValueType ProvidedType;

        /// Default value
        ProvidedType default_value;
        
        /// Provided values
        std::vector<ProvidedType> values;

         /// Construct values
        WithDefaultValue(const std::initializer_list<ProvidedType>& values, const ProvidedType& defval=ProvidedType()): default_value(defval), values(values) {}
        
        /// Construct values from iterator
        template <typename Iterator>
        explicit WithDefaultValue(const Iterator& begin, const Iterator& end, const ProvidedType& defval=ProvidedType()): default_value(defval), values(begin, end) {}
        
       /// Construct default value
        WithDefaultValue(const ProvidedType& defval=ProvidedType()): default_value(defval) {}

        /**
         * Access value.
         * \param n value index
         * \return reference to the value
         */
        ProvidedType& operator[](size_t n) {
            size_t n0 = values.size();
            if (n > n0) {
                values.resize(n);
                for (size_t i = n0; i != n; ++i) values[i] = default_value;
            }
            return values[n];
        }

        /**
         * Change number of values
         * \param n number of values
         */
        void resize(size_t n) {
            size_t n0 = values.size();
            values.resize(n);
            if (n > n0) {
                for (size_t i = n0; i != n; ++i) values[i] = default_value;
            }
        }
        
        /**
         * Get number of values
         * \return number of values
         */
        virtual size_t size() const {
            return values.size();
        }
        
        /**
         * Get provided value.
         * @return provided value
         */
        virtual ProvidedType operator()(size_t n, _ExtraParams...) const {
            if (n > values.size()) return default_value;
            return values[n];
        }
    };

    /**
     * Implementation of one value provider class which holds value inside (in value field) and operator() return its held value.
     *
     * Its value is optional and can throw exception if value was not assigned before request to it.
     *
     * It ignores eventual extra parameters.
     */
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of provided value.
        typedef ValueType ProvidedType;

        /// Provided value.
        std::vector<ProvidedType> values;

        /// Reset value to be uninitialized.
        void invalidate() { values.clear(); }

        /**
         * Throw NoValue exception if the index is wrong.
         * \param n value index
         */
        void ensureIndex(size_t n) const {
            if (n >= values.size()) throw NoValue(NAME);
        }

        /// Construct value
        explicit WithValue(const ProvidedType& value): values({value}) {}

        /// Construct value
        explicit WithValue(ProvidedType&& value): values({value}) {}

        /// Construct values
        WithValue(const std::initializer_list<ProvidedType>& values): values(values) {}
        
        /// Construct values from iterator
        template <typename Iterator>
        explicit WithValue(const Iterator& begin, const Iterator& end): values(begin, end) {}
        
        /// Create empty boost::optional value.
        WithValue() {}
        
        /**
         * Access value value.
         * \param n value index
         * \return reference to the value
         */
        ProvidedType& operator[](size_t n) {
            if (n > values.size())  throw BadInput(NAME, "Wrong value index");
            return values[n];
        }

        /**
         * Add new value
         * \param val new value
         */
        void push_back(const ProvidedType& value) {
            values.push_back(value);
        }
        
        /**
         * Add new value
         * \param val new value
         */
        template <typename... Args>
        void emplate_back(Args&&... args) {
            values.emplace_back(std::forward<Args>(args)...);
        }
        
        /**
         * Get number of values
         * \return number of values
         */
        virtual size_t size() const {
            return values.size();
        }
        
        /**
         * Get provided value.
         * \return provided value
         * \param n value index
         * \throw NoValue if value is empty boost::optional
         */
        virtual ProvidedType operator()(size_t n, _ExtraParams...) const {
            ensureIndex(n);
            return values[n];
        }
    };

    /**
     * Implementation of one value provider class which delegates all operator() calls to external functor.
     */
    struct Delegate: public PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(size_t, _ExtraParams...)> {

        typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(size_t, _ExtraParams...)> Base;
        
        std::function<size_t()> sizeGetter;

        /**
        * Create delegate provider
        * \param functor delegate functor
        */
        template<typename Functor, typename Sizer>
        Delegate(Functor functor, Sizer sizer): Base(functor), sizeGetter(sizer) {}

        /**
         * Create delegate provider
         * \param object object of class with delegate method
         * \param member delegate member method
         * \param sizer class member returning number of the elements
         */
        template<typename ClassType, typename MemberType>
        Delegate(ClassType* object, MemberType member, size_t (ClassType::*sizer)()const): Base(object, member),
            sizeGetter([object, sizer]() { return (object->*sizer)(); }) {}
            
        /**
         * Create delegate provider
         * \param object object of class with delegate method
         * \param member delegate member method
         * \param sizer class member returning number of the elements
         */
        template<typename ClassType, typename MemberType>
        Delegate(ClassType* object, MemberType member, size_t (ClassType::*sizer)()): Base(object, member),
            sizeGetter([object, sizer]() { return (object->*sizer)(); }) {}

        virtual size_t size() const {
            return sizeGetter();
        }
           
    };

    /// Used by receivers as const value provider, see Receiver::setConst
    typedef WithValue ConstProviderType;

};

/**
 * Specialization which implements provider class which provides values in mesh points and uses interpolation.
 */
template <typename PropertyT, typename SpaceT, typename... _ExtraParams>
struct ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >: public FieldProvider<typename PropertyAtSpace<PropertyT, SpaceT>::ValueType, SpaceT, _ExtraParams...> {

    static constexpr const char* NAME = PropertyT::NAME;
    virtual const char* name() const { return NAME; }

    static_assert(!std::is_same<SpaceT, void>::value,
                  "Providers for fields properties require SpaceT. Use ProviderFor<propertyTag, SpaceT>, where SpaceT is one of the class defined in plask/geometry/space.h.");

    typedef typename PropertyAtSpace<PropertyT, SpaceT>::ValueType ValueType;

    /// Type of provided value.
    typedef typename FieldProvider<ValueType, SpaceT>::ProvidedType ProvidedType;

    /**
     * Template for implementation of field provider class which holds vector of values and mesh inside.
     * operator() call plask::interpolate.
     * @tparam MeshPtrType type of pointer (shared_ptr or unique_ptr) to mesh which is used for calculation and which describe places of data points
     */
    template <typename MeshPtrType>
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of mesh pointer
        typedef MeshPtrType MeshPointerType;

        /// Type of provided value.
        typedef ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::ProvidedType ProvidedType;

        /// Provided value. Values in points described by this->mesh.
        ProvidedType values;

      protected:

        /// Mesh which describes in which points there are this->values.
        MeshPtrType mesh_ptr;

        /// Default interpolation method
        InterpolationMethod default_interpolation;

      public:

        /**
         * Get mesh.
         * @return @c *mesh_ptr
         */
        auto getMesh() -> decltype(*mesh_ptr) { return *mesh_ptr; }

        /**
         * Get mesh (const).
         * @return @c *mesh_ptr
         */
        auto getMesh() const -> decltype(*mesh_ptr) { return *mesh_ptr; }

        /**
         * Set a new Mesh.
         * \param mesh_p pointer to the new mesh
         */
        void setMesh(MeshPtrType mesh_p) {
            if (mesh_ptr) mesh_ptr->changedDisconnectMethod(this, &ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::WithValue<MeshPtrType>::onMeshChange);
            mesh_ptr = mesh_p;
            mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /// Reset values to uninitialized state (nullptr data).
        void invalidate() { values.reset(); }

        /// Reserve memory in values using mesh size.
        void allocate() { values.reset(mesh_ptr->size); }

        /**
         * Check if this has value / is initialized.
         * @return @c true only if this is initialized (has value)
         */
        bool hasValue() const { return values.data() != nullptr; }

        /// Throw NoValue exception if value is not initialized
        void ensureHasValue() const {
            if (!hasValue()) throw NoValue(name());
        }

        /**
         * Check if this has value of the right size.
         * @return @c true only if this is initialized (has value)
         */
        bool hasCorrectValue() const { return hasValue() && values.size() == mesh_ptr->size(); }

        /// Throw NoValue exception if value is not initialized and BadMesh exception if the mesh and values sizes mismatch
        void ensureHasCorrectValue() const {
            if (!hasValue()) throw NoValue(name());
            if (values.size() != mesh_ptr->size())
                throw BadMesh("Provider::WithValue", "Mesh size (%2%) and values size (%1%) do not match", values.size(), mesh_ptr->size());
        }

        /**
         * This method is called when mesh was changed.
         * It's just call invalidate()
         * @param evt information about mesh changes
         */
        void onMeshChange(const Mesh::Event& evt) {
            this->invalidate();
        }


        /**
         * @param values provided value, values in points describe by this->mesh.
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation default interpolation method for this provider
         */
        explicit WithValue(ProvidedType values, const MeshPtrType& mesh_ptr = nullptr, InterpolationMethod default_interpolation = INTERPOLATION_LINEAR)
            : values(values), mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation type of interpolation to use as default
         */
        explicit WithValue(MeshPtrType mesh_ptr = nullptr, const InterpolationMethod& default_interpolation = INTERPOLATION_LINEAR)
            : mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        ~WithValue() {
            if (mesh_ptr) mesh_ptr->changedDisconnectMethod(this, &ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * Get provided value in points described by this->mesh.
         * @return provided value in points described by this->mesh
         */
        ProvidedType& operator()() {
            ensureHasCorrectValue();
            return values;
        }

        /**
         * Get provided value in points described by this->mesh.
         * @return provided value in points described by this->mesh
         */
        const ProvidedType& operator()() const {
            ensureHasCorrectValue();
            return values;
        }

        /**
         * Calculate interpolated values using plask::interpolate.
         * @param dst_mesh set of requested points
         * @param method method which should be use to do interpolation
         * @return values in points described by mesh @a dst_mesh
         */
        virtual ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod method = INTERPOLATION_DEFAULT) const {
            ensureHasCorrectValue();
            if (method == INTERPOLATION_DEFAULT) method = default_interpolation;
            return interpolate(*mesh_ptr, values, dst_mesh, method);
        }
    };

    /**
     * Implementation of field provider class which delegates all operator() calls to external functor.
     */
    typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod method)> Delegate;

    /**
     * Return same value in all points.
     *
     * Used by receivers as const value provider, see Receiver::setConst
     *
     * It ignores extra parameters.
     */
    struct ConstProviderType: public ProviderFor<PropertyT, SpaceT> {

        typedef ProviderImpl<PropertyT, FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::ProvidedType ProvidedType;

        /// Provided value
        ValueType value;

        //ConstProviderType(const ValueT& value): value(value) {}

        /**
         * Constructor which delegate all parameters to value constructor.
         * @param params ValueT constructor parameters, forwarded to value
         */
        template<typename ...Args>
        ConstProviderType(Args&&... params): value(std::forward<Args>(params)...) {}

        /**
         * @return copy of value for each point in dst_mesh, ignore interpolation method
         */
        virtual ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod) const {
            return ProvidedType(dst_mesh.size(), value);
        }
    };
};

/**
 * Specialization which implements provider class which provides multiple values in mesh points and uses interpolation.
 */
template <typename PropertyT, typename SpaceT, typename... _ExtraParams>
struct ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >: public MultiFieldProvider<typename PropertyAtSpace<PropertyT, SpaceT>::ValueType, SpaceT, _ExtraParams...> {

    static constexpr const char* NAME = PropertyT::NAME;
    virtual const char* name() const { return NAME; }

    static_assert(!std::is_same<SpaceT, void>::value,
                  "Providers for fields properties require SpaceT. Use ProviderFor<propertyTag, SpaceT>, where SpaceT is one of the class defined in plask/geometry/space.h.");

    typedef typename PropertyAtSpace<PropertyT, SpaceT>::ValueType ValueType;

    /// Type of provided value.
    typedef typename MultiFieldProvider<ValueType, SpaceT>::ProvidedType ProvidedType;

    /**
     * Template for implementation of field provider class which holds vector of values and mesh inside.
     * operator() call plask::interpolate.
     * @tparam MeshPtrType type of pointer (shared_ptr or unique_ptr) to mesh which is used for calculation and which describe places of data points
     */
    template <typename MeshPtrType>
    struct WithValue: public ProviderFor<PropertyT, SpaceT> {

        /// Type of mesh pointer
        typedef MeshPtrType MeshPointerType;

        /// Type of provided value.
        typedef ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::ProvidedType ProvidedType;

        /// Provided values. Values in points described by this->mesh.
        std::vector<ProvidedType> values;

      protected:

        /// Mesh which describes in which points there are this->values.
        MeshPtrType mesh_ptr;

        /// Default interpolation method
        InterpolationMethod default_interpolation;

      public:

        /**
         * Get mesh.
         * @return @c *mesh_ptr
         */
        auto getMesh() -> decltype(*mesh_ptr) { return *mesh_ptr; }

        /**
         * Get mesh (const).
         * @return @c *mesh_ptr
         */
        auto getMesh() const -> decltype(*mesh_ptr) { return *mesh_ptr; }

        /**
         * Set a new Mesh.
         * \param mesh_p pointer to the new mesh
         */
        void setMesh(MeshPtrType mesh_p) {
            if (mesh_ptr) mesh_ptr->changedDisconnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::WithValue<MeshPtrType>::onMeshChange);
            mesh_ptr = mesh_p;
            mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /// Reset values to uninitialized state (nullptr data).
        void invalidate() { values.clear(); }

        /** Reserve memory for n values using mesh size.
         * \param num number of values
         */
        void allocate(size_t num) {
            values.resize(num);
            for (auto& val: values) val.reset(mesh_ptr->size);
        }

        /**
         * Add new value
         * \param val new value
         */
        void push_back(const ProvidedType& value) {
            values.push_back(value);
        }
        
        /**
         * Get number of values
         * \return number of values
         */
        virtual size_t size() const {
            return values.size();
        }
        
        /**
         * Check if this has values of the right size.
         * \param n value index
         * \return \c true only if this is initialized (has value)
         */
        bool hasCorrectValue(size_t n) const {
            return n < values.size() && values[n].size() = mesh_ptr->size();
        }

        /**
         * Check if this has values of the right size.
         * @return @c true only if this is initialized (has value)
         */
        bool hasCorrectValues() const {
            if (values.size() == 0) return false;
            for (const auto& val: values)
                if (val.size() != mesh_ptr->size()) return false;
            return true;
        }

        /**
         * Throw NoValue exception if value is not initialized and BadMesh exception if the mesh and values sizes mismatch
         * \param n value index
         */
        void ensureHasCorrectValue(size_t n) const {
            if (n >= values.size()) throw NoValue(name());
            if (values[n].size() != mesh_ptr->size())
                    throw BadMesh("Provider::WithValue", "Mesh size (%2%) and values[%3%] size (%1%) do not match", values.size(), mesh_ptr->size(), n);
        }

        /// Throw NoValue exception if value is not initialized and BadMesh exception if the mesh and values sizes mismatch
        void ensureHasCorrectValues() const {
            if (values.size() == 0) throw NoValue(name());
            for (size_t i = 0; i != values.size(); ++i)
                if (values[i].size() != mesh_ptr->size())
                    throw BadMesh("Provider::WithValue", "Mesh size (%2%) and values[%3%] size (%1%) do not match", values.size(), mesh_ptr->size(), i);
        }

        /**
         * This method is called when mesh was changed.
         * It's just call invalidate()
         * @param evt information about mesh changes
         */
        void onMeshChange(const Mesh::Event& evt) {
            this->invalidate();
        }


        /**
         * @param values single (first) provided value
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation default interpolation method for this provider
         */
        explicit WithValue(ProvidedType values, const MeshPtrType& mesh_ptr = nullptr, InterpolationMethod default_interpolation = INTERPOLATION_LINEAR)
            : values({values}), mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * @param values provided values list
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation default interpolation method for this provider
         */
        explicit WithValue(const std::initializer_list<ProvidedType>& values, const MeshPtrType& mesh_ptr = nullptr, InterpolationMethod default_interpolation = INTERPOLATION_LINEAR)
            : values(values), mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * @param begin,end iterators to range to construct values from
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation default interpolation method for this provider
         */
        template <typename Iterator>
        explicit WithValue(const Iterator& begin, const Iterator& end, const MeshPtrType& mesh_ptr = nullptr, InterpolationMethod default_interpolation = INTERPOLATION_LINEAR)
            : values(begin, end), mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * @param mesh_ptr pointer to mesh which describes in which points there are this->values
         * @param default_interpolation type of interpolation to use as default
         */
        explicit WithValue(MeshPtrType mesh_ptr = nullptr, const InterpolationMethod& default_interpolation = INTERPOLATION_LINEAR)
            : mesh_ptr(mesh_ptr), default_interpolation(default_interpolation) {
            if (mesh_ptr) mesh_ptr->changedConnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::WithValue<MeshPtrType>::onMeshChange);
        }

        ~WithValue() {
            if (mesh_ptr) mesh_ptr->changedDisconnectMethod(this, &ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...> >::WithValue<MeshPtrType>::onMeshChange);
        }

        /**
         * Get provided value in points described by this->mesh.
         * \param n value index
         * \return provided value in points described by this->mesh
         */
        ProvidedType& operator()(size_t n) {
            ensureHasCorrectValue(n);
            return values[n];
        }

        /**
         * Get provided value in points described by this->mesh.
         * \param n value index
         * \return provided value in points described by this->mesh
         */
        const ProvidedType& operator()(size_t n) const {
            ensureHasCorrectValue(n);
            return values[n];
        }

        /**
         * Calculate interpolated values using plask::interpolate.
         * \param n value index
         * \param dst_mesh set of requested points
         * \param method method which should be use to do interpolation
         * \return values in points described by mesh \a dst_mesh
         */
        virtual ProvidedType operator()(size_t n, const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod method = INTERPOLATION_DEFAULT) const {
            ensureHasCorrectValue(n);
            if (method == INTERPOLATION_DEFAULT) method = default_interpolation;
            return interpolate(*mesh_ptr, values[n], dst_mesh, method);
        }
    };

    /**
     * Implementation of field provider class which delegates all operator() calls to external functor.
     */
    struct Delegate: public PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(size_t n, const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod method)> {
        
        typedef PolymorphicDelegateProvider<ProviderFor<PropertyT, SpaceT>, ProvidedType(size_t n, const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod method)> Base;
        
        std::function<size_t()> sizeGetter;

        /**
         * Create delegate provider
         * \param functor delegate functor
         * \param sizer functor returning number of the elements
         */
        template<typename Functor, typename Sizer>
        Delegate(Functor functor, Sizer sizer): Base(functor), sizeGetter(sizer) {}

        /**
         * Create delegate provider
         * \param object object of class with delegate method
         * \param member delegate member method
         * \param sizer functor returning number of the elements
         */
        template<typename ClassType, typename MemberType, typename Sizer>
        Delegate(ClassType* object, MemberType member, Sizer sizer): Base(object, member), sizeGetter(sizer) {}
            
        /**
         * Create delegate provider
         * \param object object of class with delegate method
         * \param member delegate member method
         * \param sizer class member returning number of the elements
         */
        template<typename ClassType, typename MemberType>
        Delegate(ClassType* object, MemberType member, size_t (ClassType::*sizer)()const): Base(object, member),
            sizeGetter([object, sizer]() { return (object->*sizer)(); }) {}
            
        /**
         * Create delegate provider
         * \param object object of class with delegate method
         * \param member delegate member method
         * \param sizer class member returning number of the elements
         */
        template<typename ClassType, typename MemberType>
        Delegate(ClassType* object, MemberType member, size_t (ClassType::*sizer)()): Base(object, member),
            sizeGetter([object, sizer]() { return (object->*sizer)(); }) {}
            
        virtual size_t size() const {
            return sizeGetter();
        }
           
    };

    /**
     * Return same value in all points.
     *
     * Used by receivers as const value provider, see Receiver::setConst
     *
     * It ignores extra parameters.
     */
    struct ConstProviderType: public ProviderFor<PropertyT, SpaceT> {

        typedef ProviderImpl<PropertyT, MULTI_FIELD_PROPERTY, SpaceT, VariadicTemplateTypesHolder<_ExtraParams...>>::ProvidedType ProvidedType;

        /// Provided value
        ValueType value;

        //ConstProviderType(const ValueT& value): value(value) {}

        /**
         * Constructor which delegate all parameters to value constructor.
         * @param params ValueT constructor parameters, forwarded to value
         */
        template<typename ...Args>
        ConstProviderType(Args&&... params): value(std::forward<Args>(params)...) {}

        /**
         * @return copy of value for each point in dst_mesh, ignore interpolation method
         */
        virtual ProvidedType operator()(size_t, const MeshD<SpaceT::DIM>& dst_mesh, _ExtraParams..., InterpolationMethod) const {
            return ProvidedType(dst_mesh.size(), value);
        }
        
        virtual size_t size() const {
            return 1;
        }
    };
};

}   // namespace plask

#endif // PLASK__PROVIDERFOR_H
