#ifndef PLASK__GEOMETRY_BORDER_H
#define PLASK__GEOMETRY_BORDER_H

#include "../material/db.h"
#include "primitives.h"

#include "../log/log.h"

namespace plask {

namespace border {

/**
 * Base, abstract for all classes which describe what do with points outside geometry in calculation space.
 */
struct PLASK_API Strategy {

    /// Enum holding strategy types
    enum Type {
        DEFAULT,
        SIMPLE,
        EXTEND,
        PERIODIC,
        MIRROR
    };

    virtual ~Strategy() {}

    /// \return strategy type
    virtual Type type() const = 0;

    /**
     * Apply strategy to given point @p p.
     * @param[in] bbox_lo, bbox_hi coordinates of geometry object bounding box in startegy working direction
     * @param[in,out] p coordinate of point in startegy working direction, it's (must be) lower than @p bbox_lo, this method can move this point
     * @param[out] result_material optionaly, this method can assign to it material which should be used
     * \param[in] opposite strategy at opposite side (if known)
     */
    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const = 0;

    /**
     * Apply strategy to given point @p p.
     * @param[in] bbox_lo, bbox_hi coordinates of geometry object bounding box in startegy working direction
     * @param[in,out] p coordinate of point in startegy working direction, it's (must be) higher than @p bbox_hi, this method can move this point
     * @param[out] result_material optionaly, this method can assign to it material which should be used
     * \param[in] opposite strategy at opposite side (if known)
     */
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const = 0;

    /**
     * Check if this strategy can move point p to place outside bounding box.
     *
     * Default implementation return @c false.
     * @return @c true if this strategy can move point p to place outside bounding box,
     *         @c false if this strategy always move point p to bounding box or doesn't move p at all
     */
    virtual bool canMoveOutsideBoundingBox() const;

    /*
     * Check if this strategy can coexists on opposite side with oppositeStrategy.
     */
    /*virtual bool canCoexistsWith(const Strategy& oppositeStrategy) const {
        return true;
    }*/

    /*void ensureCanCoexists(const Strategy& oppositeStrategy) const {
        if (!canCoexistsWith(oppositeStrategy) || !oppositeStrategy.canCoexistsWith(*this))
            throw Exception("Border strategies \"{0}\" and \"{1}\" can't be used on opposite sides.", this->str(), oppositeStrategy.str());
    }*/

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
     *
     * Throw excption if @p str not describe strategy.
     * @param str string which represent strategy, one of: "null", "periodic", "extend", "mirror", or material.
     * @param materialsDB source of materials, typically material database, used to get material
     * @return created strategy
     */
    static Strategy* fromStr(const std::string& str, const MaterialsDB& materialsDB = MaterialsDB::getDefault());

    /**
     * Create new strategy described by string @p str.
     *
     * Throw excption if @p str not describe strategy.
     * @param str string which represent strategy, one of: "null", "periodic", "extend", "mirror", or material.
     * @param materialsDB source of materials, typically material database, used to get material
     * @return created strategy manged by unique_ptr, same as <code>std::unique_ptr<Strategy>(fromStr(str, materialsDB))</code>
     */
    static std::unique_ptr<Strategy> fromStrUnique(const std::string& str, const MaterialsDB& materialsDB = MaterialsDB::getDefault()) {
        return std::unique_ptr<Strategy>(fromStr(str, materialsDB));
    }
};

/**
 * Base class for all universal strategies.
 *
 * Universal strategies form subset of strategies, and could be required in some context.
 */
struct PLASK_API UniversalStrategy: public Strategy {
    virtual UniversalStrategy* clone() const override = 0;
};

/**
 * Strategy which does nothing.
 */
struct PLASK_API Null: public UniversalStrategy {
    virtual Type type() const override { return DEFAULT; }
    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual Null* clone() const override;
    virtual std::string str() const override;
};

/**
 * Strategy which assigns constant material.
 */
struct PLASK_API SimpleMaterial: public UniversalStrategy {

    /**
     * Material which will be assigned to result_material by apply method.
     */
    shared_ptr<Material> material;

    /**
     * Construct SimpleMaterial strategy wich use given material.
     * @param material material which will be assigned to result_material by apply method
     */
    SimpleMaterial(const shared_ptr<Material>& material): material(material) {}

    virtual Type type() const override { return SIMPLE; }

    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;

    virtual SimpleMaterial* clone() const override;

    virtual std::string str() const override;

};

/**
 * Strategy which moves point p to nearest border.
 */
struct PLASK_API Extend: public UniversalStrategy {

    virtual Type type() const override { return EXTEND; }

    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;

    virtual Extend* clone() const override;

    virtual std::string str() const override;

};

/**
 * Strategy which moves point p by multiple of (bbox_hi - bbox_lo) to be in range [bbox_lo, bbox_hi].
 */
struct PLASK_API Periodic: public Strategy {

    virtual Type type() const override { return PERIODIC; }

    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;

    virtual Periodic* clone() const override;

    virtual std::string str() const override;

    /*virtual bool canCoexistsWith(const Strategy& oppositeStrategy) const {
        return oppositeStrategy.type() == Strategy::PERIODIC;
    }*/
};

struct PLASK_API Mirror: public Strategy {

    virtual Type type() const override { return MIRROR; }

    virtual void applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;
    virtual void applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material, const Strategy* opposite) const override;

    virtual bool canMoveOutsideBoundingBox() const override;

    virtual Mirror* clone() const override;

    virtual std::string str() const override;

};

/**
 * Held border strategy with given type and:
 * - delegate apply methods to held strategy,
 * - allow to assing strategy to self (using operator=).
 * @tparam direction held strategy working direction (coordinate of vector component)
 * @tparam StrategyType (base) type of held strategy, typically Strategy or UniversalStrategy
 */
template <int direction, typename StrategyType = Strategy>
class PLASK_API StrategyHolder {

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

    std::string str() const { return strategy->str(); }

    StrategyHolder<direction, StrategyType>& operator=(const StrategyType& strategy) { setStrategy(strategy); return *this; }

    template <int adirection>
    StrategyHolder<direction, StrategyType>& operator=(const StrategyHolder<adirection, StrategyType>& strategyHolder) {
        setStrategy(strategyHolder.getStrategy());
        return *this;
    }

    /*inline void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const {
        strategy->apply(bbox_lo, bbox_hi, p, result_material);
    }*/

    template <int dims>
    inline void applyLo(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material, const Strategy* opposite) const {
        strategy->applyLo(bbox.lower[direction], bbox.upper[direction], p[direction], result_material, opposite);
    }

    template <int dims>
    inline void applyHi(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material, const Strategy* opposite) const {
        strategy->applyHi(bbox.lower[direction], bbox.upper[direction], p[direction], result_material, opposite);
    }

    template <int dims>
    inline void applyIfLo(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material, const Strategy* opposite) const {
        if (p[direction] < bbox.lower[direction])
            applyLo(bbox, p, result_material, opposite);
    }

    template <int dims>
    inline void applyIfHi(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material, const Strategy* opposite) const {
        if (p[direction] > bbox.upper[direction])
            applyHi(bbox, p, result_material, opposite);
    }

};

/**
 * Held pairs of strategies (for lo and hi band) with given type.
 * @tparam direction hold strategy working direction (coordinate of vector component)
 * @tparam StrategyType (base) type of held strategies, typically Strategy or UniversalStrategy
 */
template <int direction, typename StrategyType = Strategy>
class PLASK_API StrategyPairHolder {
    /// lo and hi strategy
    StrategyHolder<direction, StrategyType> strategy_lo, strategy_hi;

    /// If true strategies calling order is: hi, lo
    bool reverseCallingOrder;

    void setOrder(const StrategyType& strategy_lo, const StrategyType& strategy_hi) {
        if ((strategy_lo.type() == Strategy::PERIODIC || strategy_hi.type() == Strategy::PERIODIC) &&
             strategy_lo.type() != Strategy::MIRROR && strategy_hi.type() != Strategy::MIRROR &&
             strategy_lo.type() != strategy_hi.type()
           ) writelog(LOG_WARNING, "Periodic and non-periodic borders strategies used on opposite sides of one direction.");
        //strategy_lo.ensureCanCoexists(strategy_hi);
        if (strategy_hi.canMoveOutsideBoundingBox()) {
            if (strategy_lo.canMoveOutsideBoundingBox())
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
     * @param setNewHi if @c true new hi strategy will be set, in other case new lo strategy will be set
     * @param strategy new strategy value
     */
    void set(bool setNewHi, const StrategyType& strategy) {
        if (setNewHi) setHi(strategy); else setLo(strategy);
    }

    /**
     * Get lo or hi strategy.
     * @param _getHi if @c true hi strategy will be returned, in other case lo strategy will be returned
     * @return lo or hi strategy, depends from @p _getHi value
     */
    const StrategyType& get(bool _getHi) const {
        return _getHi ? getHi() : getLo();
    }

    template <int dims>
    inline void apply(const typename Primitive<dims>::Box& bbox, Vec<dims, double>& p, shared_ptr<Material>& result_material) const {
        if (reverseCallingOrder) {
            strategy_hi.applyIfHi(bbox, p, result_material, &strategy_lo.getStrategy());
            if (result_material) return;
            strategy_lo.applyIfLo(bbox, p, result_material, &strategy_hi.getStrategy());
        } else {
            strategy_lo.applyIfLo(bbox, p, result_material, &strategy_hi.getStrategy());
            if (result_material) return;
            strategy_hi.applyIfHi(bbox, p, result_material, &strategy_lo.getStrategy());
        }
    }

    bool isSymmetric() const { return strategy_lo.type() == Strategy::MIRROR || strategy_hi.type() == Strategy::MIRROR; }

    bool isPeriodic() const { return strategy_lo.type() == Strategy::PERIODIC && strategy_hi.type() == Strategy::PERIODIC; }
};

}   // namespace border

}   // namespace plask

#endif // PLASK__GEOMETRY_BORDER_H
