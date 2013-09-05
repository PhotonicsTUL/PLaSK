#ifndef PLASK__PROVIDER_H
#define PLASK__PROVIDER_H

/** @file
This file contains base classes for providers and receivers.
@see @ref providers


@page providers Providers and receivers

@section providers_about About provider-receiver mechanism
This page describe providers and receivers mechanism, which allow for data exchange between solvers.

Provider is an object which has type derived from plask::Provider and provide some value(s)
(has operator() which return provided value(s)).
It also has set of listeners which are inform about changes of provided data.

Receiver is an object of class, which is typically connected with provider and allow for reading value(s)provided by it
(has operator() which return provided value(s)).

Each type of provider has corresponding type of receiver (see plask::Receiver),
and only provider and receiver witch corresponding types can be connected.

@section providers_in_solvers Using providers and receivers in solvers
Each solver should have one provider class field for each physical property which it want to make
available for other solvers and reports and it also should have one receiver field for each physical
property which value it wants to know (needs for calculations).
Most providers are classes obtain by using plask::ProviderFor template.

See @ref solvers_writing for more details and examples.

An example of using providers and receivers in solvers can be found in description of plask::Temperature.

@section providers_writing Writing new providers and receivers types

@subsection providers_writing_easy Easy (half-automatic) way
The easiest way to create new provider and corresponding receiver types is to write physical property
tag class and use it to specialize plask::ProviderFor and plask::ReceiverFor templates.

Physical property tag class is an class which only has static fields and methods and typedefs which describe
physical property. It can be easy obtain by subclass instantiation of one of templates:
- plask::Property — allows to obtain all possible physical properties tags classes, but require many parameters (not recommended);
- plask::SingleValueProperty — allows to obtain tags for properties described by one value (typically one scalar), require only one parameter - type of provided value;
- plask::FieldProperty — allows to obtain tags for properties described by values in points described by mesh, require only one parameter - type of provided value;
- plask::ScalarFieldProperty — equals to plask::FieldProperty\<double\>, doesn't require any parameters,
- plask::CustomFieldProperty - allows to obtain tags for properties described by values in points described by mesh and allows to points two types of provided values (first is used in 2D space and second is used in 3D spaces),
- plask::VectorFieldProperty — allows to obtain tags for properties described by values in points described by mesh uses two types of provided values: Vec<2, X> in 2D and Vec<3, X> in 3D, where X is given type.

This subclass can include static field and methods:
- <code>static constexpr const char* NAME = "(lowercase) name of the property"</code>,
- <code>static inline ValueType getDefaultValue() { return ...; }</code> -
  construct and return initial/default instance of varible with provided type (default implementation returns <code>ValueType()</code>),
- <code>static inline ValueType2D getDefaultValue2D() { return ...; }</code> and <code>static inline ValueType3D getDefaultValue3D() { return ...; }</code> -
  like above for properties which uses different types in 2D and 3D spaces (e.g. plask::VectorFieldProperty),
- <code>static const ValueType2D& value3Dto2D(const ValueType3D& v) { return ...; }</code> (convert v to 2D) and
  <code>static const ValueType3D& value2Dto3D(const ValueType2D& v) { return ...; }</code> (convert v to 3D) -
  convert values between 2D and 3D spaces, only for properties which uses different types in 2D and 3D spaces, used by filters.

Extra templates parameters can be passed to each property tag class template described above.
This parameters are types of extra arguments required by provider to obtain value.

Both templates plask::ProviderFor and plask::ReceiverFor may take two parameters:
- first is physical property tag and it's required,
- second is type of space and it's required (and allowed) only for fields properties.

plask::ProviderFor class cannot be used directly, but one must declare it using some specialized class within the plask::ProviderFor namespace.
E.g. \b plask::ProviderFor<MyProperty>::WithValue. The specialized class \b WithValue specifies how the provided values can be obtained.
You can choose from the following options:
- \b WithValue (available only for plask::SingleValueProperty) — the value is stored in the provider itself.
  It can be assigned a value just like any class member field. Mind that the additional property parameters are ignored by this provider!
- \b WithDefaultValue (available only for plask::SingleValueProperty) — similar to \b WithValue, however it always has some value.
  Use it if there is always some sensible default value for the provided quantity, even before any calculations have been performed.
  Again, the additional property parameters are ignored by this provider!
- \b Delegate (available for all properties) — the solver needs to contain the method that computes the provided value (field or scalar) on demand.
  This provider requires the pointer to both the solver containing it and the this method as its constructor arguments. See \ref solvers_writing_details
  for an example.

Example:
@code
// Physical property tag class for something.
struct MyProperty: public plask::SingleValueProperty<double> {
    static constexpr const char* NAME = "my property"; // use lowercase here
};

// Base type for MyProperty provider.
typedef plask::ProviderFor<MyProperty> MyPropertyProvider;

// Type for MyProperty receiver class.
typedef plask::ReceiverFor<MyProperty> MyPropertyReceiver;

// ...
// Usage example:
MyPropertyProvider::WithValue provider;
MyPropertyReceiver receiver;
receiver.setProvider(provider);       // connect
provider = 2.0;             // set some value to provider
assert(receiver() == 2.0);  // test the received value

// .........

// Physical property tag with additional parameter of type 'int'.
struct ParamProperty: public plask::SingleValueProperty<double, int> {
    static constexpr const char* NAME = "property with parameter"; // use lowercase here
};

// ...
// Usage example:
plask::ProviderFor<ParamProperty>::Delegate provider2([](int i) { return 2.0*i; });
plask::ReceiverFor<ParamProperty> receiver2;
receiver2.setProvider(provider2);      // connect
assert(receiver2(3) == 6.0); // test the received value

@endcode


@subsection providers_writing_manual Flexible (manual) way

The (described @ref providers_writing_easy "above") method of creating providers and receivers should be sufficient for most cases. However, there is also
a harder but more flexible approach than using plask::ProviderFor and plask::ReceiverFor templates. You can write your own provider class which
inherits from plask::Provider and has operator(), which for some parameters (depending on your choice) returns the provided value.

Receiver class for your provider still may be very easy obtained by plask::Receiver template. This template requires only one parameter: the type of the provider.
You can use it directly or as a base class for your receiver.

Example:
@code
// Provider type which multiple its argument by value
struct ScalerProvider: public plask::Provider {

    double scale;

    ScalerProvider(double scale): scale(scale) {}

    double operator()(double param) const {
        return scale * param;
    }
};

// Receiver corresponding to ScalerProvider
typedef Receiver<ScalerProvider> ScalerReceiver;
// or class ScalerReceiver: public Receiver<ScalerProvider> { ... };

// ...
// Usage example:
ScalerProvider sp(2.0);
ScalerReceiver sr;
sr.setProvider(sp);               // connect
assert(sr(3.0) == 6.0); // test the received value
@endcode
*/

#include <set>
#include <vector>
#include <functional>   // std::function
#include <type_traits>  // std::is_same
#include <boost/optional.hpp>
#include <boost/signals2.hpp>


#include "../exceptions.h"
#include "../utils/stl.h"
#include "../mesh/mesh.h"
#include "../mesh/interpolation.h"

namespace plask {

/**
 * Base class for all Providers.
 *
 * It implements listener (observer) pattern (can be observed by Receiver).
 *
 * Subclasses should only implement operator()(...) which returns provided value, or throw NoValue exception.
 * Receiver (for given provider type) can be easy implemented by inherit Receiver class template.
 *
 * @see @ref providers
 */
struct Provider {

    static constexpr const char* NAME = "undefined";
    virtual const char* name() const { return NAME; }

    Provider & operator=(const Provider&) = delete;
    Provider(const Provider&) = delete;
    Provider() = default;

    /**
     * Signal called when providers value has been changed or provider is being deleted.
     * Only in second case second parameter is @c true.
     */
    boost::signals2::signal<void(Provider&, bool)> changed;

    /// Call onDisconnect for all listeners in listeners set.
    virtual ~Provider() {
        changed(*this, true);
    }

    /**
     * Call onChange for all listeners.
     * Should be called after change of value represented by this provider.
     */
    void fireChanged() {
        changed(*this, false);
    }

};

/**
 * Common non-template base for all receivers.
 * This class is usefult for metaprogramming and also can be used for holding pointers to receivers.
 */
struct ReceiverBase {

    /// The reason of change of provider value
    enum class ChangeReason {
        EVENT_DELETE,     ///< this receiver is being deleted
        VALUE,      ///< value of provider has just been changed
        PROVIDER    ///< provider has just been exchanged to another one
    };

    virtual ~ReceiverBase() {}
};

/**
 * Base class for all Receivers.
 *
 * Implement listener (observer) pattern (is listener for providers).
 * Delegate all operator() calling to provider.
 *
 * For most providers types, Receiver type can be defined as: <code>Receiver<ProviderClass>;</code>
 * (where <code>ProviderClass</code> is type of provider class)
 *
 * @tparam ProviderT type of provider, can has defined ProviderT::ConstProviderType to reciver setConst method work.
 *
 * @see @ref providers
 */
template <typename ProviderT>
class Receiver: public ReceiverBase {

    boost::signals2::connection providerConnection;

    /// Is @c true only if data provides by provider was changed after previous value retrieval.
    bool _changed;

protected:

    /// Is @c true only if provider is private and will be deleted by this receiver.
    bool _hasPrivateProvider;

public:

    typedef Receiver<ProviderT> Base;

    /// Name of provider.
    static constexpr const char* PROVIDER_NAME = ProviderT::NAME;
    virtual const char* providerName() const { return PROVIDER_NAME; }

    /// Signal called when provider value or provider was changed (called by onChange)
    boost::signals2::signal<void(ReceiverBase& src, ChangeReason reason)> providerValueChanged;

    Receiver& operator=(const Receiver&) = delete;
    Receiver(const Receiver&) = delete;

    /// Type of the corresponding provider
    typedef ProviderT ProviderType;

    /// Pointer to connected provider. Can be nullptr if no provider is connected.
    ProviderT* provider;

    /// Construct Receiver without connected provider and with set changed flag.
    Receiver(): _changed(true), _hasPrivateProvider(false), provider(0) {}

    /**
     * Check if data provides by provider was changed after previous value retrieval.
     * @return @c true only if data provides by provider was changed after previous value retrieval
     */
    bool changed() const { return _changed; }

    /**
     * Set change flag. This flag is set if data provides by provider was changed after previous value retrieval.
     * @param new_value new value for changed flag
     */
    void changed(bool new_value) { _changed = new_value; }

    /// Destructor. Disconnect from provider.
    virtual ~Receiver() {
        providerConnection.disconnect();
        if (_hasPrivateProvider) {
            delete this->provider;
            this->provider = nullptr;
        }
        fireChanged(ChangeReason::EVENT_DELETE);
    }

    /**
     * Set change flag and call providerValueChanged with given @p reason.
     * @param reason passed to providerValueChanged signal
     */
    void fireChanged(ChangeReason reason) {
        _changed = true;
        providerValueChanged(*this, reason);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, can be @c nullptr to only disconnect from current provider.
     * @param newProviderIsPrivate @c true only if @p provider is private for this and will be deleted by this receiver
     */
    void setProvider(ProviderT* provider, bool newProviderIsPrivate = false) {
        if (this->provider == provider) {
            this->_hasPrivateProvider = newProviderIsPrivate;
            return;
        }
        providerConnection.disconnect();
        if (_hasPrivateProvider) delete this->provider;
        if (provider) providerConnection = provider->changed.connect(
                    [&](Provider& which, bool isDeleted) {
                        if (isDeleted) {
                            providerConnection.disconnect();    //TODO do we need this line?
                            this->provider = 0;
                        }
                        this->fireChanged(isDeleted ? ChangeReason::PROVIDER : ChangeReason::VALUE);
                    });
        this->provider = provider;
        this->_hasPrivateProvider = newProviderIsPrivate;
        this->fireChanged(ChangeReason::PROVIDER);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider
     */
    void setProvider(ProviderT &provider) {
        setProvider(&provider);
    }

    /**
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, will be deleted by this receiver
     */
    void setProvider(std::unique_ptr<ProviderT>&& provider) {
        setProvider(provider->release(), true);
    }

    /*
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider, can be @c nullptr to only disconnect from current provider.
     */
    /*void operator=(ProviderT *provider) {
        setProvider(provider);
    }*/

    /*
     * Change provider. If new provider is different from current one then changed flag is set.
     * @param provider new provider
     */
    /*void operator=(ProviderT &provider) {
        setProvider(&provider);
    }*/

    /**
     * Current provider getter.
     * @return current provider or @c nullptr if there is no connected provider
     */
    ProviderT* getProvider() { return provider; }

    /**
     * Current provider getter.
     * @return current provider or @c nullptr if there is no connected provider
     */
    const ProviderT* getProvider() const { return provider; }

    /// \return true if there is any provider connected
    bool hasProvider() {
        return provider;
    }

    /// @throw NoProvider when provider is not available
    void ensureHasProvider() const {
        if (provider == nullptr)
            throw NoProvider(providerName());
    }

    /**
     * Get value from provider using its operator().
     * @return value from provider
     * @throw NoProvider when provider is not available
     * @throw NoValue when provider can't give value (is uninitialized, etc.)
     */
    template<typename ...Args> auto
    operator()(const Args&... params) const -> decltype((*provider)(params...)) {
        beforeGetValue();
        return (*provider)(params...);
    }

    /**
     * Get value from provider using its operator().
     * If value can't be gotten (there is no provider or provider can't give value) empty optional is returned.
     * @return value from provider or empty optional if value couldn't be got
     */
    template<typename ...Args> auto
    optional(const Args&... params) const -> boost::optional<decltype((*provider)(params...))> {
        try {
            return boost::optional<decltype((*provider)(params...))>(this->operator()(params...));
        } catch (std::exception&) {
            const_cast<Receiver*>(this)->_changed = false; // unless anything changes, next call to optional will return the same
            return boost::optional<decltype((*provider)(params...))>();
        }
    }

    /**
     * Set provider for this to provider of constant.
     *
     * Use ProviderT::ConstProviderType as provider of const type.
     * @param constProviderConstructorArgs parameters passed to ProviderT::ConstProviderType constructor
     */
    template <typename ...ConstProviderConstructorArgs>
    void setConstValue(ConstProviderConstructorArgs&&... constProviderConstructorArgs) {
        setProvider(new typename ProviderT::ConstProviderType(std::forward<ConstProviderConstructorArgs>(constProviderConstructorArgs)...), true);
    }

    /**
     * Connect a method to changed signal.
     * @param obj, method slot to connect, object and it's method
     * @param at specifies where the slot should be connected:
     *  - boost::signals2::at_front indicates that the slot will be connected at the front of the list or group of slots
     *  - boost::signals2::at_back (default) indicates that the slot will be connected at the back of the list or group of slots
     */
    template <typename ClassT, typename methodT>
    boost::signals2::connection changedConnectMethod(ClassT* obj, methodT method, boost::signals2::connect_position at=boost::signals2::at_back) {
        return providerValueChanged.connect(boost::bind(method, obj, _1, _2), at);
    }

    /// Disconnect a method from changed signal
    template <typename ClassT, typename methodT>
    void changedDisconnectMethod(ClassT* obj, methodT method) {
        providerValueChanged.disconnect(boost::bind(method, obj, _1, _2));
    }

protected:

    /**
     * Check if value can be read and throw exception if not.
     * Set changed flag to false.
     *
     * Subclass can call this just before reading value.
     *
     * @throw NoProvider when provider is not available
     */
    void beforeGetValue() const {
        ensureHasProvider();
        const_cast<Receiver*>(this)->_changed = false;
    }

};

/**
 * Instantiation of this template is abstract base class for provider which provide one value (for example one double).
 * @tparam ValueT type of provided value
 * @tparam ArgsT type of arguments required by provider (optional)
 */
template <typename ValueT, typename... ArgsT>
struct SingleValueProvider: public Provider {

    static constexpr const char* NAME = "undefined value";
    virtual const char* name() const { return NAME; }

    /// Type of provided value.
    typedef ValueT ProvidedType;

    /**
     * Provided value getter.
     * @return provided value
     */
    virtual ProvidedType operator()(ArgsT...) const = 0;

};

/**
 * Instantiation of this template is abstract base class for provider which provide multiple values (for example one double).
 * @tparam ValueT type of provided value
 * @tparam ArgsT type of arguments required by provider (optional)
 */
template <typename ValueT, typename... ArgsT>
struct MultiValueProvider: public Provider {

    static constexpr const char* NAME = "undefined value";
    virtual const char* name() const { return NAME; }

    /// Type of provided value.
    typedef ValueT ProvidedType;

    /**
     * Provided value getter.
     * @return provided value
     */
    virtual ProvidedType operator()(size_t num, ArgsT...) const = 0;

    /**
     * Get number of values
     * \return number of values
     */
    virtual size_t size() const = 0;

};

//TODO typedef for SingleValueReceiver (GCC 4.7 needed)

/**
 * Instantiation of this template is abstract base class for provider class which provide values in points described by mesh
 * and use interpolation.
 */
template <typename ValueT, typename SpaceT, typename... ExtraArgs>
struct FieldProvider: public Provider {

    static constexpr const char* NAME = "undefined field";
    virtual const char* name() const { return NAME; }

    /// Type of value provided by this (returned by operator()).
    typedef DataVector<const ValueT> ProvidedType;

    /**
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    /**
     * Call this->operator()(dst_mesh, DEFAULT).
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, ExtraArgs... extra_args) const {
        return this->operator()(dst_mesh, extra_args..., INTERPOLATION_DEFAULT);
    }

    /**
     * Call this->operator()(*dst_mesh, extra_args..., method).
     * @param dst_mesh set of requested points, given in shared_ptr
     * @param extra_args additional provider arguments
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, ExtraArgs... extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        return this->operator()(*dst_mesh, extra_args..., method);
    }

    /**
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments, given in tuple
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(const MeshD<SpaceT::DIM>& dst_mesh, std::tuple<ExtraArgs...>&& extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        typedef std::tuple<ExtraArgs...> Tuple;
        return apply_tuple(dst_mesh, method, std::forward<Tuple>(extra_args), make_seq_indices<0, sizeof...(ExtraArgs)>{});
    }

    /**
     * @param dst_mesh set of requested points, given in shared_ptr
     * @param extra_args additional provider arguments, given in tuple
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, std::tuple<ExtraArgs...> extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        return this->operator()(*dst_mesh, extra_args, method);
    }

private:
    template <typename T,  template <std::size_t...> class I, std::size_t... Indices>
    inline ProvidedType apply_tuple(const MeshD<SpaceT::DIM>& dst_mesh, InterpolationMethod method, T&& t, I<Indices...>) {
      return this->operator()(dst_mesh, std::get<Indices>(std::forward<T>(t))..., method);
    }

};

/**
 * Instantiation of this template is abstract base class for provider class which provide values in points described by mesh
 * and use interpolation.
 */
template <typename ValueT, typename SpaceT, typename... ExtraArgs>
struct MultiFieldProvider: public Provider {

    static constexpr const char* NAME = "undefined field";
    virtual const char* name() const { return NAME; }

    /// Type of value provided by this (returned by operator()).
    typedef DataVector<const ValueT> ProvidedType;

    /**
     * Get number of values
     * \return number of values
     */
    virtual size_t size() const = 0;

    /**
     * @param num number of the value
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual ProvidedType operator()(size_t num, const MeshD<SpaceT::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    /**
     * Call this->operator()(dst_mesh, DEFAULT).
     * @param num number of the value
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(size_t num, const MeshD<SpaceT::DIM>& dst_mesh, ExtraArgs... extra_args) const {
        return this->operator()(num, dst_mesh, extra_args..., INTERPOLATION_DEFAULT);
    }

    /**
     * Call this->operator()(*dst_mesh, extra_args..., method).
     * @param num number of the value
     * @param dst_mesh set of requested points, given in shared_ptr
     * @param extra_args additional provider arguments
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(size_t num, shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, ExtraArgs... extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        return this->operator()(num, *dst_mesh, extra_args..., method);
    }

    /**
     * @param num number of the value
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments, given in tuple
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(size_t num, const MeshD<SpaceT::DIM>& dst_mesh, std::tuple<ExtraArgs...>&& extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        typedef std::tuple<ExtraArgs...> Tuple;
        return apply_tuple(num, dst_mesh, method, std::forward<Tuple>(extra_args), make_seq_indices<0, sizeof...(ExtraArgs)>{});
    }

    /**
     * @param num number of the value
     * @param dst_mesh set of requested points, given in shared_ptr
     * @param extra_args additional provider arguments, given in tuple
     * @param method method which should be use to do interpolation
     * @return values in points describe by mesh @a dst_mesh
     */
    inline ProvidedType operator()(size_t num, shared_ptr<const MeshD<SpaceT::DIM>> dst_mesh, std::tuple<ExtraArgs...> extra_args, InterpolationMethod method = INTERPOLATION_DEFAULT) const {
        return this->operator()(num, *dst_mesh, extra_args, method);
    }

private:
    template <typename T,  template <std::size_t...> class I, std::size_t... Indices>
    inline ProvidedType apply_tuple(size_t num, const MeshD<SpaceT::DIM>& dst_mesh, InterpolationMethod method, T&& t, I<Indices...>) {
      return this->operator()(num, dst_mesh, std::get<Indices>(std::forward<T>(t))..., method);
    }

};


template<typename _Signature> struct DelegateProvider;

/**
 * Template of class which is good base class for providers which delegate calls of operator() to external functor
 * (function or method).
 * @tparam _Res(_ArgTypes...) functor signature (result and arguments types)
 */
template<typename _Res, typename... _ArgTypes>
struct DelegateProvider<_Res(_ArgTypes...)>: public Provider {

    /// Held external functor.
    std::function<_Res(_ArgTypes...)> valueGetter;

    /**
     * Initialize valueGetter using given params.
     * @param params parameters for valueGetter constructor
     */
    template<typename ...Args>
    DelegateProvider<_Res(_ArgTypes...)>(Args&&... params)
    : valueGetter(std::forward<Args>(params)...) {
    }

    /**
     * Call functor held by valueGetter.
     * @param params parameters for functor held by valueGetter
     * @return value returned by functor held by valueGetter
     */
    virtual _Res operator()(_ArgTypes&&... params) const {
        return valueGetter(std::forward<_ArgTypes>(params)...);
    }
};

template<typename _BaseClass, typename _Signature> struct PolymorphicDelegateProvider;

/**
 * Template of class which is a good base class for providers which delegate calls of operator() to external functor
 * (function or method).
 * @tparam _Res(_ArgTypes...) functor signature (result and arguments types)
 */
template<typename _BaseClass, typename _Res, typename... _ArgTypes>
struct PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>: public _BaseClass {

    /// Held external functor.
    std::function<_Res(_ArgTypes...)> valueGetter;

    /**
     * Create delegate provider
     * \param functor delegate functor
     */
    template<typename Functor>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(Functor functor)
        : valueGetter(functor)
    {}

    /**
     * Create delegate provider
     * \param object object of class with delegate method
     * \param member delegate member method
     */
    template<typename ClassType, typename MemberType>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(ClassType* object, MemberType member)
        : valueGetter(
          [object, member](_ArgTypes&&... params) {
              return (object->*member)(std::forward<_ArgTypes>(params)...);
          })
    {}

    /**
     * Initialize valueGetter using given parameters.
     * @param params parameters for valueGetter constructor
     */
    template<typename ...Args>
    PolymorphicDelegateProvider<_BaseClass, _Res(_ArgTypes...)>(Args&&... params)
    : valueGetter(std::forward<Args>(params)...) {
    }

    /**
     * Call functor held by valueGetter.
     * @param params parameters for functor held by valueGetter
     * @return value returned by functor held by valueGetter
     */
    _Res operator()(_ArgTypes... params) const {
        return valueGetter(std::forward<_ArgTypes>(params)...);
    }
};

}; //namespace plask

#endif  //PLASK__PROVIDER_H
