#ifndef PLASK__GEOMETRY_BORDER_H
#define PLASK__GEOMETRY_BORDER_H

#include "../material/db.h"
#include "primitives.h"

namespace plask {

namespace border {

/**
 * Base, abstract for all classes which describe what do with points outside geometry in calculation space.
 */
struct Strategy {

    /**
     * Apply strategy to given point @p p.
     * @param bbox_lo[in], bbox_hi[in] coordinates of geometry element bounding box in startegy working direction
     * @param p[in,out] coordinate of point in startegy working direction, it's lower than @p bbox_lo or higher than @p bbox_hi, this method can move this point
     * @param result_material[out] optionaly, this method can assign to it material which should be used
     */
    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const = 0;

    /**
     * Check if this strategy can move point p to place outside bounding box.
     *
     * Default implementation return @c false.
     * @return @c true if this strategy can move point p to place outside bounding box,
     *         @c false if this strategy always move point p to bounding box or doesn't move p at all
     */
    virtual bool canMoveOutsideBoundingBox() const;

    /**
     * Clone this strategy.
     * @return strategy identical to this one, constructed using operator new
     */
    virtual Strategy* clone() const = 0;

    /**
     * Get string representation of this strategy.
     * @return string representation of this strategy
     */
    virtual std::string str() const = 0;

    /**
     * Create new strategy (using operator new) described by string @p str.
     * @param str string which represent strategy, one of: "null", "periodic", "extend", "mirror", or material.
     * @param materialsDB material database used to get material
     * @return created strategy
     */
    static Strategy* fromStr(const std::string& str, const MaterialsDB& materialsDB = MaterialsDB::getDefault());
};

/**
 * Base class for all universal strategies.
 *
 * Universal strategies form subset of strategies, and could be required in some context.
 */
struct UniversalStrategy: public Strategy {
    virtual UniversalStrategy* clone() const = 0;
};

/**
 * Strategy which does nothing.
 */
struct Null: public UniversalStrategy {
    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;
    virtual Null* clone() const;
    virtual std::string str() const;
};

/**
 * Strategy which assigns constant material.
 */
struct SimpleMaterial: public UniversalStrategy {

    /**
     * Material which will be assigned to result_material by apply method.
     */
    shared_ptr<Material> material;

    /**
     * Construct SimpleMaterial strategy wich use given material.
     * @param material material which will be assigned to result_material by apply method
     */
    SimpleMaterial(const shared_ptr<Material>& material): material(material) {}

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual SimpleMaterial* clone() const;

    virtual std::string str() const;

};

/**
 * Strategy which moves point p to nearest border.
 */
struct Extend: public UniversalStrategy {

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual Extend* clone() const;

    virtual std::string str() const;

};

/**
 * Strategy which moves point p by multiple of (bbox_hi - bbox_lo) to be in range [bbox_lo, bbox_hi].
 */
struct Periodic: public Strategy {

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual Periodic* clone() const;

    virtual std::string str() const;
};

struct Mirror: public Strategy {

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual bool canMoveOutsideBoundingBox() const;

    virtual Mirror* clone() const;

    virtual std::string str() const;

};

/**
 * Hold border strategy with given type and:
 * - delegate apply methods to holded strategy,
 * - allow to assing strategy to self (using operator=).
 * @tparam direction holded strategy working direction (coordinate of vector component)
 * @tpatam StrategyType (base) type of holded strategy, typically Strategy or UniversalStrategy
 */
template <int direction, typename StrategyType = Strategy>
class StrategyHolder {

    StrategyType* strategy;

public:
    StrategyHolder(const StrategyType& strategy = Null()): strategy(strategy.clone()) {}

    template <int adirection>
    StrategyHolder(const StrategyHolder<adirection, StrategyType>& strategyHolder): strategy(strategyHolder.strategy->clone()) {}

    ~StrategyHolder() { delete strategy; }

    const StrategyType& getStrategy() const { return *strategy; }

    void setStrategy(const StrategyType& strategy) {
        if (this->strategy == &strategy) return;    //self-assigment protect
        delete this->strategy;
        this->strategy = strategy.clone();
    }

    StrategyHolder<direction, StrategyType>& operator=(const StrategyType& strategy) { setStrategy(strategy); return *this; }

    template <int adirection>
    StrategyHolder<direction, StrategyType>& operator=(const StrategyHolder<adirection, StrategyType>& strategyHolder) {
        setStrategy(strategyHolder.getStrategy());
        return *this;
    }

    inline void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const {
        strategy->apply(bbox_lo, bbox_hi, p, result_material);
    }

    template <int dims>
    inline void apply(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material) const {
        apply(bbox.lower.components[direction], bbox.upper.components[direction],
              p.components[direction], result_material);
    }

    template <int dims>
    inline void applyIfLo(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material) const {
        if (p.components[direction] < bbox.lower.components[direction])
            apply(bbox, p, result_material);
    }

    template <int dims>
    inline void applyIfHi(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material) const {
        if (p.components[direction] > bbox.upper.components[direction])
            apply(bbox, p, result_material);
    }

};

/**
 * Hold pairs of strategies (for lo and hi band) with given type.
 * @tparam direction holded strategy working direction (coordinate of vector component)
 * @tpatam StrategyType (base) type of holded strategies, typically Strategy or UniversalStrategy
 */
template <int direction, typename StrategyType = Strategy>
class StrategyPairHolder {
    /// lo and hi strategy
    StrategyHolder<direction, StrategyType> strategy_lo, strategy_hi;

    /// if true strategies calling order are: hi, lo
    bool reverseCallingOrder;

    void setOrder(const StrategyType& strategy_lo, const StrategyType& strategy_hi) {
        if (strategy_lo.canMoveOutsideBoundingBox()) {
            if (strategy_hi.canMoveOutsideBoundingBox())
                throw Exception("Border strategies on both sides can move point outside bounding box.");
            reverseCallingOrder = true;
        } else
            reverseCallingOrder = false;
    }

public:

    StrategyPairHolder(): reverseCallingOrder(false) {}

    void setStrategies(const StrategyType& strategy_lo, const StrategyType& strategy_hi) {
        setOrder(strategy_lo, strategy_hi);
        this->strategy_lo = strategy_lo;
        this->strategy_hi = strategy_hi;
    }

    void setBoth(const StrategyType& s) { setStrategies(s, s); }

    void setLo(const StrategyType& strategy_lo) {
        setOrder(strategy_lo, getHi());
        this->strategy_lo = strategy_lo;
    }

    const StrategyType& getLo() const {
        return strategy_lo.getStrategy();
    }

    void setHi(const StrategyType& strategy_hi) {
        setOrder(getLo(), strategy_hi);
        this->strategy_hi = strategy_hi;
    }

    const StrategyType& getHi() const {
        return strategy_hi.getStrategy();
    }

    /**
     * Set lo or hi strategy.
     * @param setNewHi if @true new hi strategy will be set, in other case new lo strategy will be set
     * @param strategy new strategy value
     */
    void set(bool setNewHi, const StrategyType& strategy) {
        if (setNewHi) setHi(strategy); else setLo(strategy);
    }

    /**
     * Get lo or hi strategy.
     * @param _getHi if @true hi strategy will be returned, in other case lo strategy will be returned
     * @param lo or hi strategy, depends from @p _getHi value
     */
    const StrategyType& get(bool _getHi) const {
        return _getHi ? getHi() : getLo();
    }

    template <int dims>
    inline void apply(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material) const {
        if (reverseCallingOrder) {
            strategy_hi.applyIfHi(bbox, p, result_material);
            if (result_material) return;
            strategy_lo.applyIfLo(bbox, p, result_material);
        } else {
            strategy_lo.applyIfLo(bbox, p, result_material);
            if (result_material) return;
            strategy_hi.applyIfHi(bbox, p, result_material);
        }
    }
};

}   // namespace border

}   // namespace plask

#endif // PLASK__GEOMETRY_BORDER_H
