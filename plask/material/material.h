#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file contains base classes for materials and material database class.
*/

#include <string>
#include <map>
#include <vector>
#include <functional>
#include <tuple>
#include <type_traits>

#include "../math.h"
#include "../memory.h"
#include "../exceptions.h"
#include "../phys/constants.h"
#include "../phys/functions.h"
#include "../vector/tensor2.h"
#include "../vector/tensor3.h"
#include "../parallel.h"

namespace plask {

/// Global Python lock
extern PLASK_API OmpNestLock material_omp_lock;

/**
 * Get group in periodic table of given @p object.
 * @param objectName name of object
 * @return group of object with given name @p objectName or 0 if given object is not known
 */
int objectGroup(const std::string& objectName);

struct MaterialsDB;

/**
 * Represent material, its physical properties.
 */
struct PLASK_API Material {

    /// Dopting specification type
    enum DopingAmountType {
        NO_DOPING,              ///< no dopant
        DOPANT_CONCENTRATION,   ///< doping concentration
        CARRIER_CONCENTRATION   ///< carrier concentration
    };

    /// Material kind
    enum Kind {
        NONE,           ///< no material or air
        SEMICONDUCTOR,  ///< semiconductor
        OXIDE,          ///< oxide
        DIELECTRIC,     ///< other dielectric
        METAL,          ///< metal
        LIQUID_CRYSTAL, ///< liquid crystal
        MIXED           ///< artificial mix of several materials with averaged properties
    };

    /// Material conductivity type.
    enum ConductivityType {
        CONDUCTIVITY_N,           ///< n-type semiconductor
        CONDUCTIVITY_I,           ///< i-type semiconductor
        CONDUCTIVITY_P,           ///< p-type semiconductor
        CONDUCTIVITY_OTHER,       ///< other conductivity
        CONDUCTIVITY_UNDETERMINED ///< undetermined conductivity (e.g. a mixed material)
    };

    /**
     * Type for material composition.
     */
    typedef std::map<std::string, double> Composition;


  private:

    /// Check if material can be constructed with composition.
    template <typename MaterialType>
    struct is_with_composition {
        static const bool value =
            std::is_constructible<MaterialType, Composition>::value ||
            std::is_constructible<MaterialType, Composition, DopingAmountType, double>::value;
    };

    /// Check if material can be constructed with dopant.
    template <typename MaterialType>
    struct is_with_dopant {
        static const bool value =
            std::is_constructible<MaterialType, DopingAmountType, double>::value ||
            std::is_constructible<MaterialType, Composition, DopingAmountType, double>::value;
    };

    friend struct MaterialsDB;

  public:

    /**
     * Parameters of material, information about: name, composition and dopant.
     *
     * It stores all information which are represented by material string,
     * but without precision lossing (amounts are stored in doubles).
     *
     * Can be obtained either from string (see parse(std::string)) or material (see getParameters()).
     */
    struct PLASK_API Parameters {

        /// short (without composition and doping amounts) name of material
        /// only for simple material(?)
        std::string name;

        Composition composition;

        std::string dopant_name;

        double dopantAmount;

        Material::DopingAmountType dopantAmountType;

        explicit Parameters(const std::string& name): name(name), dopantAmountType(NO_DOPING) {}



        bool isSimple() const { return composition.empty(); }

        bool hasDopant() const { return dopantAmountType != NO_DOPING; }
    };

    /**
     * Helper class for easy constructing string representations of complex materials.
     *
     * Typically this is used to implement str() method.
     *
     * Example:
     * @code
     * double Al = 0.6;
     * double Mg = 0.1;
     * std::string str = StringBuilder("Al", Al)("Ga")("N").dopant("Mg", Mg);
     * //str is "Al(0.6)GaN:Mg=0.1"
     * @endcode
     */
    struct PLASK_API StringBuilder {

        /// Part of name which has been already built
        std::stringstream str;

        /**
         * Cast to string operator.
         * @return part of name which has been already built
         */
        operator std::string() const { return str.str(); }

        /**
         * Append name of object (without ammount) to built string.
         * @param objectName name of object to add
         * @return *this
         */
        StringBuilder& operator()(const std::string& objectName) { str << objectName; return *this; }

        /**
         * Construct builder and append name of object (without ammount) to built string.
         * @param objectName name of object to add
         */
        StringBuilder(const std::string& objectName) {
            this->operator ()(objectName);
        }

        /**
         * Append name of object (with ammount) to built string.
         * @param objectName name of object to add
         * @param ammount ammount of added object
         * @return *this
         */
        StringBuilder& operator()(const std::string& objectName, double ammount);

        /**
         * Construct builder and append name of object (with ammount) to built string.
         * @param objectName name of object to add
         * @param ammount ammount of added object
         */
        StringBuilder(const std::string& objectName, double ammount) {
            this->operator ()(objectName, ammount);
        }

        /**
         * Append information about doping to built string.
         * @param dopantName name of dopant
         * @param dopantConcentration dopant concentration
         * @return built material name
         */
        std::string dopant(const std::string& dopantName, double dopantConcentration);

        /**
         * Append information about doping to built string.
         * @param dopantName name of dopant
         * @param n_or_p 'n' or 'p'
         * @param carrierConcentration carrier concentration
         * @return built material name
         */
        std::string dopant(const std::string& dopantName, char n_or_p, double carrierConcentration);

    };

    /**
     * Parse composition object from [begin, end) string.
     * @param begin begin of string, will be increased to point to potential next composition object or end (if parsed composition object was last one)
     * @param end points just after last charcter of string, must be: begin < end
     * @return parsed object name and ammount (NaN if there was no information about ammount)
     */
    static std::pair<std::string, double> firstCompositionObject(const char*& begin, const char* end);

    /**
     * Change NaN-s in material composition to calculated amounts.
     *
     * Throw exception if it is impossible to complete given composition.
     * @param composition amounts of objects composition with NaN on position for which amounts has not been taken
     * @return complate composition, for example for ("Al", 0.7), ("Ga", NaN), ("N", NaN) result is ("Al", 0.7), ("Ga", 0.3), ("N", 1.0)
     */
    static Composition completeComposition(const Composition& composition);

    /**
     * Parse composition from string, or string fragment.
     *
     * Throws exception in case of parsing errors.
     * @param begin, end [begin, end) string or range in string, for example "Al(0.7)GaN"
     * @return parsed composition, can be not complate, for "Al(0.7)GaN" result is ("Al", 0.7), ("Ga", NaN), ("N", NaN)
     * @see @ref completeComposition
     */
    static Composition parseComposition(const char* begin, const char* end);

    /**
     * Parse composition from string.
     *
     * Throws exception in case of parsing errors.
     * @param composition_str composition string, objects and amounts, for example "Al(0.7)GaN"
     * @return parsed composition, can be not complate, for "Al(0.7)GaN" result is ("Al", 0.7), ("Ga", NaN), ("N", NaN)
     * @see @ref completeComposition
     */
    static Composition parseComposition(const std::string& composition_str);

    /**
     * Parse information about dopant from string.
     *
     * Throws exception in case of parsing errors.
     * @param[in] begin, end [begin, end) string or range in string
     * @param[out] dopant_elem_name, doping_amount_type, doping_amount parsed values
     */
    static void parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount);

    /**
     * Parse information about dopant from string.
     *
     * Throws exception in case of parsing errors.
     * @param[in] dopant string to parse
     * @param[out] dopant_elem_name, doping_amount_type, doping_amount parsed values
     */
    static void parseDopant(const std::string& dopant, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount);

    /**
     * Split object name to objects.
     * @param begin, end [begin, end) string or range in string, for example "AlGaN"
     * @return vector of parsed objects (for "AlGaN" result is ["Al", "Ga", "N"])
     * @throw MaterialParseException when name is ill-formatted
     */
    static std::vector<std::string> parseObjectsNames(const char* begin, const char* end);

    /**
     * Split object name to objects.
     * @param allNames all objects names, for example "AlGaN"
     * @return vector of parsed objects (for "AlGaN" result is ["Al", "Ga", "N"])
     * @throw MaterialParseException when name is ill-formated
     */
    static std::vector<std::string> parseObjectsNames(const std::string& allNames);

    /// Do nothing.
    virtual ~Material() {}

    /**
     * Get short (without composition and doping amounts) name of material.
     * @return material name
     */
    virtual std::string name() const = 0;

    /**
     * Get full (with composition and doping amounts) name of material.
     *
     * Default implementation returns name, which is fine only for simple materials.
     * @return material name with information about composition and doping
     * @see NameBuilder
     */
    virtual std::string str() const;

    /// @return material kind
    virtual Kind kind() const = 0;

    /**
     * Get lattice constant [A].
     * @param T temperature [K]
     * @param x lattice parameter [-]
     * @return lattice constant [A]
     */
    virtual double lattC(double T, char x) const;

    /**
     * Get energy gap Eg [eV]
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in the Brillouin zone [-]
     * @return energy gap Eg [eV]
     */
    virtual double Eg(double T, double e=0., char point='G') const;

    /**
     * Get conduction band level CB [eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in the Brillouin zone [-]
     * @return conduction band level CB [eV]
     */
    virtual double CB(double T, double e=0., char point='G') const;

    /**
     * Get valence band level VB[eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @param hole hole type ('H'eavy or 'L'ight) [-]
     * @return valence band level VB[eV]
     */
    virtual double VB(double T, double e=0., char point='G', char hole='H') const;

    /**
     * Get split-off energy Dso [eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @return split-off energy Dso [eV]
     */
    virtual double Dso(double T, double e=0.) const;

    /**
     * Get split-off mass Mso [\f$m_0\f$].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @return split-off mass Mso [\f$m_0\f$]
     */
    virtual double Mso(double T, double e=0.) const;

    /**
     * Get electron effective mass Me in in-plane (lateral) and cross-plane (vertical) direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @return electron effective mass Me [\f$m_0\f$]
     */
    virtual Tensor2<double> Me(double T, double e=0., char point='G') const;

    /**
     * Get heavy hole effective mass Mhh in in-plane (lateral) and cross-plane (vertical) direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @return heavy hole effective mass Mhh [\f$m_0\f$]
     */
    virtual Tensor2<double> Mhh(double T, double e=0.) const;

    /**
     * Get light hole effective mass Mlh in in-plane (lateral) and cross-plane (vertical) direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @return light hole effective mass Mlh [\f$m_0\f$]
     */
    virtual Tensor2<double> Mlh(double T, double e=0.) const;

    /**
     * Get hole effective mass Mh in in-plane (lateral) and cross-plane (vertical) direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @return hole effective mass Mh [\f$m_0\f$]
     */
    virtual Tensor2<double> Mh(double T, double e=0.) const;

    /**
     * Get hydrostatic deformation potential for the conduction band ac [eV].
     * @param T temperature [K]
     * @return hydrostatic deformation potential for the conduction band ac [eV]
     */
    virtual double ac(double T) const;

    /**
     * Get hydrostatic deformation potential for the valence band av [eV].
     * @param T temperature [K]
     * @return hydrostatic deformation potential for the valence band av [eV]
     */
    virtual double av(double T) const;

    /**
     * Get shear deformation potential b [eV].
     * @param T temperature [K]
     * @return shear deformation potential b [eV]
     */
    virtual double b(double T) const;

    /**
     * Get shear deformation potential d [eV].
     * @param T temperature [K]
     * @return shear deformation potential d [eV]
     */
    virtual double d(double T) const;

    /**
     * Get elastic constant c11 [GPa].
     * @param T temperature [K]
     * @return elastic constant c11 [GPa]
     */
    virtual double c11(double T) const;

    /**
     * Get elastic constant c12 [GPa].
     * @param T temperature [K]
     * @return elastic constant c12 [GPa]
     */
    virtual double c12(double T) const;

    /**
     * Get elastic constant c44 [GPa].
     * @param T temperature [K]
     * @return elastic constant c44 [GPa]
     */
    virtual double c44(double T) const;

    /**
     * Get dielectric constant EpsR [-].
     * @param T temperature [K]
     * @return dielectric constant EpsR [-]
     */
    virtual double eps(double T) const;

    /**
     * Get electron affinity Chi[eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @return electron affinity Chi [eV]
     */
    virtual double chi(double T, double e=0., char point='G') const;

    /**
     * Get effective density of states in the conduction band Nc [cm^(-3)].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @return effective density of states in the conduction band Nc [cm^(-3)]
     */
    virtual double Nc(double T, double e=0., char point='G') const;

    /**
     * Get effective density of states in the valance band Nv [cm^(-3)].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @return effective density of states in the valance band Nv [cm^(-3)]
     */
    virtual double Nv(double T, double e=0., char point='G') const;

    /**
     * Get intrinsic carrier concentration Ni [cm^(-3)].
     * @param T temperature [K]
     * @return intrinsic carrier concentration Ni [cm^(-3)]
     */
    virtual double Ni(double T) const;

    /**
     * Get free carrier concentration N [cm^(-3)].
     * @param T temperature [K]
     * @return free carrier concentration N [cm^(-3)]
     */
    virtual double Nf(double T) const;

    /**
     * Get donor ionisation energy EactD [eV].
     * @param T temperature [K]
     * @return donor ionisation energy EactD [eV]
     */
    virtual double EactD(double T) const;

    /**
     * Get acceptor ionisation energy EactA [eV].
     * @param T temperature [K]
     * @return acceptor ionisation energy EactA [eV]
     */
    virtual double EactA(double T) const;

    /**
     * Get mobility in-plane (lateral) and cross-plane (vertical) direction [m^2/(V*s)].
     * @param T temperature [K]
     * @return mobility [m^2/(V*s)]
     */
    virtual Tensor2<double> mob(double T) const;

    /**
     * Get electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m].
     * @param T temperature [K]
     * @return electrical conductivity sigma [S/m]
     */
    virtual Tensor2<double> cond(double T) const;

    /**
     * Get electrical conductivity type. In semiconductors this indicates what type of carriers \a Nf refers to.
     * \return electrical conductivity type of material
     */
    virtual ConductivityType condtype() const;

    /**
     * Get monomolecular recombination coefficient A [1/s].
     * @param T temperature [K]
     * @return monomolecular recombination coefficient A [1/s]
     */
    virtual double A(double T) const;

    /**
     * Get radiative recombination coefficient B [m^3/s].
     * @param T temperature [K]
     * @return radiative recombination coefficient B [m^3/s]
     */
    virtual double B(double T) const;

    /**
     * Get Auger recombination coefficient C [m^6/s].
     * @param T temperature [K]
     * @return Auger recombination coefficient C [m^6/s]
     */
    virtual double C(double T) const;

    /**
     * Get ambipolar diffusion coefficient D [m^2/s].
     * @param T temperature [K]
     * @return ambipolar diffusion coefficient D [m^2/s]
     */
    virtual double D(double T) const;

    /**
     * Get thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction k [W/(m*K)].
     * @param T temperature [K]
     * @param h layer thickness [µm]
     * @return thermal conductivity k [W/(m*K)]
     */
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const;

    /**
     * Get density [kg/m^3].
     * @param T temperature [K]
     * @return density [kg/m^3]
     */
    virtual double dens(double T) const;

    /**
     * Get specific heat at constant pressure [J/(kg*K)].
     * @param T temperature [K]
     * @return specific heat at constant pressure [J/(kg*K)]
     */
    virtual double cp(double T) const;

    /**
     * Get refractive index Nr [-].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return refractive index Nr [-]
     */
    virtual double nr(double wl, double T, double n = 0) const;

    /**
     * Get absorption coefficient alpha [cm^(-1)].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return absorption coefficient alpha cm^(-1)]
     */
    virtual double absp(double wl, double T) const;

    /**
     * Get refractive index Nr [-].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index Nr[-]
     */
    virtual dcomplex Nr(double wl, double T, double n = 0) const;

    /**
     * Get anisotropic refractive index tensor NR [-].
     * Tensor must have the form \f$ \left[\begin{array}{ccc} n_{0} & n_{3} & 0\\ n_{4} & n_{1} & 0\\ 0 & 0 & n_{2} \end{array}\right] \f$,
     * where \f$ n_i \f$ is i-th object of the returned tuple.
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return refractive index tensor NR[-]
     */
    virtual Tensor3<dcomplex> NR(double wl, double T, double n = 0) const;

    /**
     * Check if this material is equal to @a other (checks type and uses isEqual).
     * @param other other material
     * @return @c true only if this is equal to @p other
     */
    bool operator==(const Material& other) const;

    /**
     * Check if this material is equal to @a other (checks type and uses isEqual).
     * @param other other material
     * @return @c true only if this is equal to @p other, @c false in case of other is nullptr
     */
    bool operator==(shared_ptr<const Material> other) const {
        return other ? this->operator==(other) : false;
    }

    /**
     * Check if this material is deifferent from @a other (checks type and uses isEqual).
     * @param other other material
     * @return @c true only if this is not equal to @p other
     */
    bool operator!=(const Material& other) const { return ! this->operator==(other); }

    /**
     * Check if this material is deifferent from @a other (checks type and uses isEqual).
     * @param other other material
     * @return @c true only if this is not equal to @p other, @c true in case of other is nullptr
     */
    bool operator!=(shared_ptr<const Material> other) const { return ! this->operator==(other); }

protected:

    /**
     * Check if this material is equal to @a other.
     *
     * Default implementation compares string representation of this and other.
     * For simple materials (without parameters) this should just returns true.
     * @param other other material witch has the same type as this
     * (in implementation you can safty static_cast it to type of this)
     * @return @c true only if this is equal to @p other
     */
    virtual bool isEqual(const Material& other) const;

    /**
     * Throw exception with information that method with name @p method_name is not implemented for this material.
     * @param method_name name of method which is not implemented
     */
    void throwNotImplemented(const std::string& method_name) const;

    /**
     * Throw exception with information that method with name @p method_name is not applicable for this material.
     * @param method_name name of method which is not applicable
     */
    void throwNotApplicable(const std::string& method_name) const;

};


/**
 * Base material class for all semiconductors and similar materials
 */
struct PLASK_API Semiconductor: public Material {
    static constexpr const char* NAME = "semiconductor";
    virtual std::string name() const override;
    virtual Kind kind() const override;
};

/**
 * Base material class for all metals
 */
struct PLASK_API Metal: public Material {
    static constexpr const char* NAME = "metal";
    virtual std::string name() const override;
    virtual Kind kind() const override;
    virtual double eps(double T) const override;

};

/**
 * Base material class for all oxides
 */
struct PLASK_API Oxide: public Material {
    static constexpr const char* NAME = "oxide";
    virtual std::string name() const override;
    virtual Kind kind() const override;
};

/**
 * Base material class for all dielectrics
 */
struct PLASK_API Dielectric: public Material {
    static constexpr const char* NAME = "dielectric";
    virtual std::string name() const override;
    virtual Kind kind() const override;
};

/**
 * Base material class for all liquid crystals
 */
struct PLASK_API LiquidCrystal: public Material {
    static constexpr const char* NAME = "liquid_crystal";
    virtual std::string name() const override;
    virtual Kind kind() const override;
};



/**
 * Material which consist of several real materials.
 * It calculate averages for all properties.
 *
 * Example:
 * @code
 * MixedMaterial m;
 * // mat1, mat2, mat3 are materials, 2.0, 5.0, 3.0 weights for it:
 * m.add(mat1, 2.0).add(mat2, 5.0).add(mat3, 3.0).normalizeWeights();
 * double avg_VB = m.VB(300);
 * @endcode
 */
struct PLASK_API MixedMaterial: public Material {

    /** Vector of materials and its weights. */
    std::vector < std::pair<shared_ptr<Material>,double> > materials;

    /**
      Delegate all constructor to materials vector.
      */
    template<typename ...Args>
    MixedMaterial(Args&&... params)
    : materials(std::forward<Args>(params)...) {
    }

    /**
     * Scale weights in materials vector, making sum of this weights equal to 1.
     */
    void normalizeWeights();

    /**
     * Add material with weight to materials vector.
     * @param material material to add
     * @param weight weight
     */
    MixedMaterial& add(const shared_ptr<Material>& material, double weight);

    virtual ~MixedMaterial() {}

    //Material methods implementation:
    virtual std::string name() const override;

    virtual Kind kind() const override;

    virtual double lattC(double T, char x) const override;

    virtual double Eg(double T, double e=0., char point='G') const override;

    virtual double CB(double T, double e=0., char point='G') const override;

    virtual double VB(double T, double e=0., char point='G', char hole='H') const override;

    virtual double Dso(double T, double e=0.) const override;

    virtual double Mso(double T, double e=0.) const override;

    virtual Tensor2<double> Me(double T, double e=0., char point='G') const override;

    virtual Tensor2<double> Mhh(double T, double e=0.) const override;

    virtual Tensor2<double> Mlh(double T, double e=0.) const override;

    virtual Tensor2<double> Mh(double T, double e=0.) const override;

    virtual double ac(double T) const override;

    virtual double av(double T) const override;

    virtual double b(double T) const override;

    virtual double d(double T) const override;

    virtual double c11(double T) const override;

    virtual double c12(double T) const override;

    virtual double c44(double T) const override;

    virtual double eps(double T) const override;

    virtual double chi(double T, double e=0., char point='G') const override;

    virtual double Nc(double T, double e=0., char point='G') const override;

    virtual double Nv(double T, double e=0., char point='G') const override;

    virtual double Ni(double T=0.) const override;

    virtual double Nf(double T=0.) const override;

    virtual double EactD(double T) const override;

    virtual double EactA(double T) const override;

    virtual Tensor2<double> mob(double T) const override;

    virtual Tensor2<double> cond(double T) const override;

    virtual ConductivityType condtype() const override;

    virtual double A(double T) const override;

    virtual double B(double T) const override;

    virtual double C(double T) const override;

    virtual double D(double T) const override;

    virtual Tensor2<double> thermk(double T, double h) const override;

    virtual double dens(double T) const override;

    virtual double cp(double T) const override;

    virtual double nr(double wl, double T, double n = 0.0) const override;

    virtual double absp(double wl, double T) const override;

    virtual dcomplex Nr(double wl, double T, double n = 0.0) const override;

    virtual Tensor3<dcomplex> NR(double wl, double T, double n = 0.0) const override;

private:

    /**
     * Calulate weighted sums of materials (from materials vector) properties.
     * @param f functore which calculate property value for given material
     * @return calculated sum, with the same type which return functor
     * @tparam Functor type of functor which can take const Material& argument, and return something which can be multiple by scalar, added, and assigned
     */
    template <typename Functor>
    auto avg(Functor f) const -> typename std::remove_cv<decltype(f(*((const Material*)0)))>::type {
        typename std::remove_cv<decltype(f(*((const Material*)0)))>::type w_sum = 0.;
        for (auto& p: materials) {
            w_sum += std::get<1>(p) * f(*std::get<0>(p));
        }
        return w_sum;
    }

    /**
     * Calulate weighted sums of materials (from materials vector) properties.
     * @param f functore which calculate property value for given material
     * @return calculated sum, with the same type which return functor
     * @tparam Functor type of functor which can take const Material& argument, and return something which can be multiple by scalar, added, and assigned
     */
    template <typename Functor>
    auto avg_pairs(Functor f) const -> Tensor2<double> {
        Tensor2<double> w_sum(0., 0.);
        for (auto& p: materials) {
            Tensor2<double> m = f(*std::get<0>(p));
            w_sum.c00 += std::get<1>(p) * m.c00;    //std::get<1>(p) is weight of current material
            w_sum.c11 += std::get<1>(p) * m.c11;
        }
        return w_sum;
    }

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
//TODO write or remove??
struct PLASK_API RotatedMaterial: public Material {

    shared_ptr<Material> wrapped;

};

/**
 * Empty material, which can actually be instantiated
 */
struct PLASK_API EmptyMaterial : public Material {
    virtual std::string name() const { return ""; }
    virtual Material::Kind kind() const { return Material::NONE; }
    virtual bool isEqual(const Material&) const { return true; } // all empty materials are always equal
};



} // namespace plask

#endif	//PLASK__MATERIAL_H
