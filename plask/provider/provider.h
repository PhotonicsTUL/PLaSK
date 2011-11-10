
#include <set>
#include <memory>
#include <vector>

namespace plask {

/**
 * Template for base class for all Providers.
 * Implement listener pattern.
 * @tparam 
 */
template <typename ReciverT>
struct Provider {
  
  std::set<ReciverT*> recivers;
  
  ~Provider() {
    for (std::set<ReciverT*>::iterator i = recivers.begin(); i != recivers.end(); ++i)
      i->provider = 0;
  }
  
  void add(ReciverT* reciver) {
    reciver.provider = this;
    recivers.insert(reciver);
  }
  
  void remove(ReciverT* reciver) {
    reciver.provider = 0;
    recivers.remove(reciver);
  }
  
  /**
   * Call onChange for all recivers.
   * Should be call recalculation of value represented by provider.
   */
  void fireChanged() {
    for (std::set<ReciverT*>::iterator i = recivers.begin(); i != recivers.end(); ++i)
      i->onChange();
  }
  
};

template <typename ProviderT>
struct Reciver {
  
  ProviderT* provider;
  
  ///true only if data provides by provider was changed after recent value getting
  bool changed;
  
  ~Reciver() {
    setProvider(0);
  }
  
  void setProvider(ProviderT* provider) {
    if (this->provider) provider.remove(this);
    if (provider) provider.add(this);
  }
  
  void onChange() {
    changed = true;
    //TODO callback?
  }
  
  ///@throw NoProvider when provider is not available
  void ensureHasProvider() throw (NoProvider) {
    if (!provider) throw NoProvider();	//TODO some name, maybe Provider should have virtual name or name field?
  }
  
};

template <typename ValueT> struct ValueReciver;

/**
 * Template for base class for all providers which provide one value, typically one double.
 */
template <typename ValueT>
struct ValueProvider: public Provider< ValueReciver<ValueT> > {
  
  typedef ValueT ValueType;
  
  ValueT value;
  
  ValueT& operator()() { return value; }
  
  const ValueT& operator()() const { return value; }
  
  operator ValueT& () { return value; }
  
  operator const ValueT& () const { return value; }
  
};

template <typename ValueT>
struct ValueReciver: public Reciver< ValueProvider<ValueT> > {
  
  /**
   * Get value from provider.
   * @return value from provider
   * @throw NoProvider when provider is not available
   */
  ValueT operator()() const throw (NoProvider) {
    ensureHasProvider();
    changed = false;
    return provider->value;
  }
  
};

template <typename ValueT> struct OnGridInterpolatedReciver;

/**
 * Template for base class for all providers which provide values in points describe by grid,
 * use interpolation, and has vector of data.
 */
template <typename ModuleType, typename ValueT>
struct OnGridInterpolatedProvider: public Provider< OnGridInterpolatedReciver<ValueT> > {
  
  typedef ValueT ValueType;
  
  typedef std::shared_ptr< std::vector<ValueT> > ValueVecPtr;
  
  typedef std::shared_ptr< const std::vector<ValueT> > ValueConstVecPtr;
  
  typedef ValueConstVecPtr (ModuleType::*method_ptr_t)(Grid& grid, InterpolationMethod method);
  
  ModuleType* module;
  Method_Ptr module_value_get_method;
  
  ValueVecPtr value;
  
  OnGridInterpolatedProvider(ModuleType* module, Method_Ptr module_value_get_method)
  : module(module), module_value_get_method(module_value_get_method) {
  }
  
  ValueConstVecPtr operator()(Grid& grid, InterpolationMethod method) {
    return module->*module_value_get_method(grid, method);
  }
  
};

template <typename OnGridInterpolatedProviderT>
struct OnGridInterpolatedReciver: public Reciver< OnGridInterpolatedProviderT > {
  
  /**
   * Get value from provider.
   * @return value from provider
   * @throw NoProvider when provider is not available
   */
  typename OnGridInterpolatedProviderT::ValueConstVecPtr operator()(Grid& grid, InterpolationMethod method) const throw (NoProvider) {
    ensureHasProvider();
    changed = false;
    return (*provider)(grid, method);
  }
  
};



};	//namespace plask
