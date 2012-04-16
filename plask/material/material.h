#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file includes base classes for materials and material database class.
*/

#include <string>
#include <map>
#include <vector>
#include <functional>

#include "../math.h"
#include "../memory.h"
#include <plask/config.h>   //dcomplex
#include "../exceptions.h"

#include <type_traits>

namespace plask {

/**
 * Get group in periodic table of given @p element.
 * @param elementName name of element
 * @return group of element with given name @p elementName or 0 if given element is not known
 */
int elementGroup(const std::string& elementName);

/**
 * Represent material, its physical properties.
 */
struct Material {

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
        METAL,          ///< metal
        LIQUID_CRYSTAL, ///< liquid crystal
        MIXED           ///< artificial mix of several materials with averaged properties
    };

    /**
     * Type for material composition.
     */
    typedef std::map<std::string, double> Composition;

    /// Check if material can be construct with composition.
    template <typename MaterialType>
    struct is_with_composition {
        static const bool value =
            std::is_constructible<MaterialType, Composition>::value ||
            std::is_constructible<MaterialType, Composition, DopingAmountType, double>::value;
    };

    /// Check if material can be construct with dopant.
    template <typename MaterialType>
    struct is_with_dopant {
        static const bool value =
            std::is_constructible<MaterialType, DopingAmountType, double>::value ||
            std::is_constructible<MaterialType, Composition, DopingAmountType, double>::value;
    };

    /**
     * Helper class for constructing string representations of complex materials.
     *
     * Typically used to implement str() method.
     *
     * Example:
     * @code
     * double Al = 0.6;
     * double Mg = 0.1;
     * std::string str = NameBuilder("Al", Al)("Ga")("N").doppant("Mg", Mg);
     * //str is "Al(0.6)GaN:Mg=0.1"
     * @endcode
     */
    struct NameBuilder {

        /// Part of name which has been already built
        std::string str;

        /**
         * Cast to string operator.
         * @return part of name which has been already built
         */
        operator const std::string&() const { return str; }

        /**
         * Append name of element (without ammount) to built string.
         * @param elementName name of element to add
         * @return *this
         */
        NameBuilder& operator()(const std::string& elementName) { str += elementName; return *this; }

        /**
         * Construct builder and append name of element (without ammount) to built string.
         * @param elementName name of element to add
         */
        NameBuilder(const std::string& elementName) {
            this->operator ()(elementName);
        }

        /**
         * Append name of element (with ammount) to built string.
         * @param elementName name of element to add
         * @param ammount ammount of added element
         * @return *this
         */
        NameBuilder& operator()(const std::string& elementName, double ammount);

        /**
         * Construct builder and append name of element (with ammount) to built string.
         * @param elementName name of element to add
         * @param ammount ammount of added element
         */
        NameBuilder(const std::string& elementName, double ammount) {
            this->operator ()(elementName, ammount);
        }

        /**
         * Append information about doping to built string.
         * @param dopantName name of dopant
         * @param dopantConcentration dopant concentration
         * @return built material name
         */
        const std::string& dopant(const std::string& dopantName, double dopantConcentration);

        /**
         * Append information about doping to built string.
         * @param dopantName name of dopant
         * @param n_or_p 'n' or 'p'
         * @param carrierConcentration carrier concentration
         * @return built material name
         */
        const std::string& dopant(const std::string& dopantName, char n_or_p, double carrierConcentration);

    };

    /**
     * Parse composition element from [begin, end) string.
     * @param begin begin of string, will be increased to point to potential next composition element or end (if parsed composition element was last one)
     * @param end points just after last charcter of string, must be: begin < end
     * @return parsed element name and ammount (NaN if there was no information about ammount)
     */
    static std::pair<std::string, double> getFirstCompositionElement(const char*& begin, const char* end);

    /**
     * Change NaN-s in material composition to calculated amounts.
     *
     * Throw exception if it is impossible to complete given composition.
     * @param composition amounts of elements composition with NaN on position for which amounts has not been taken
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
     * @param composition_str composition string, elements and amounts, for example "Al(0.7)GaN"
     * @return parsed composition, can be not complate, for "Al(0.7)GaN" result is ("Al", 0.7), ("Ga", NaN), ("N", NaN)
     * @see @ref completeComposition
     */
    static Composition parseComposition(const std::string& composition_str);

    /**
     * Parse information about dopant from string.
     *
     * Throws exception in case of parsing errors.
     * @param begin, end [begin, end) string or range in string
     * @param dopant_elem_name[out], doping_amount_type[out], doping_amount[out] parsed values
     */
    static void parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount);

    /**
     * Parse information about dopant from string.
     *
     * Throws exception in case of parsing errors.
     * @param dopant string to parse
     * @param dopant_elem_name[out], doping_amount_type[out], doping_amount[out] parsed values
     */
    static void parseDopant(const std::string& dopant, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount);

    /**
     * Split element name to elements.
     * @param begin, end [begin, end) string or range in string, for example "AlGaN"
     * @return vector of parsed elements (for "AlGaN" result is ["Al", "Ga", "N"])
     * @throw MaterialParseException when name is ill-formated
     */
    static std::vector<std::string> parseElementsNames(const char* begin, const char* end);

    /**
     * Split element name to elements.
     * @param allNames all elements names, for example "AlGaN"
     * @return vector of parsed elements (for "AlGaN" result is ["Al", "Ga", "N"])
     * @throw MaterialParseException when name is ill-formated
     */
    static std::vector<std::string> parseElementsNames(const std::string& allNames);

    /// Do nothing.
    virtual ~Material() {}

    /**
     * Get short (without composition and dopping amounts) name of material.
     * @return material name
     */
    virtual std::string name() const = 0;

    /**
     * Get full (with composition and dopping amounts) name of material.
     *
     * Default implementation returns name, which is fine only for simple materials.
     * @return material name with informations about composition and dopping
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
     * @param T temperature [K]
     * @param point point in the Brillouin zone [-]
     * @return energy gap Eg [eV]
     */
    virtual double Eg(double T, char point) const;

    /**
     * Get conduction band offset CBO [eV].
     * @param T temperature [K]
     * @param point point in the Brillouin zone [-]
     * @return conduction band offset CBO [eV]
     */
    virtual double CBO(double T, char point) const;

    /**
     * Get valance band offset VBO[eV].
     * @param T temperature [K]
     * @return valance band offset VBO[eV]
     */
    virtual double VBO(double T) const;

    /**
     * Get split-off energy Dso [eV].
     * @param T temperature [K]
     * @return split-off energy Dso [eV]
     */
    virtual double Dso(double T) const;

    /**
     * Get split-off mass Mso [\f$m_0\f$].
     * @param T temperature [K]
     * @return split-off mass Mso [\f$m_0\f$]
     */
    virtual double Mso(double T) const;

    /**
     * Get electron effective mass Me [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return split-off mass Mso [\f$m_0\f$]
     */
    virtual double Me(double T, char point) const;

    /**
     * Get electron effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return electron effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Me_v(double T, char point) const;

    /**
     * Get electron effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return electron effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Me_l(double T, char point) const;

    /**
     * Get heavy hole effective mass Mhh [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return heavy hole effective mass Mhh [\f$m_0\f$]
     */
    virtual double Mhh(double T, char point) const;

    /**
     * Get heavy hole effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return heavy hole effective mass Mhh [\f$m_0\f$]
     */
    virtual double Mhh_v(double T, char point) const;

    /**
     * Get heavy hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return heavy hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mhh_l(double T, char point) const;

    /**
     * Get light hole effective mass Mlh [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return light hole effective mass Mlh [\f$m_0\f$]
     */
    virtual double Mlh(double T, char point) const;

    /**
     * Get light hole effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return light hole effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Mlh_v(double T, char point) const;

    /**
     * Get light hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return light hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mlh_l(double T, char point) const;

    /**
     * Get hole effective mass Mh [\f$m_0\f$].
     * @param T temperature [K]
     * @param EqType equation type [-]
     * @return hole effective mass Mh [\f$m_0\f$]
     */
    virtual double Mh(double T, char EqType) const;

    /**
     * Get hole effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return hole effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Mh_v(double T, char point) const;

    /**
     * Get hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mh_l(double T, char point) const;

    /**
     * Get dielectric constant EpsR [-].
     * @param T temperature [K]
     * @return dielectric constant EpsR [-]
     */
    virtual double eps(double T) const;

    /**
     * Get electron affinity Chi[eV].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return electron affinity Chi [eV]
     */
    virtual double chi(double T, char point) const;

    virtual double chi(char point) const;

    /**
     * Get effective density of states in the conduction band Nc [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @param point point in Brillouin zone [-]
     * @return effective density of states in the conduction band Nc [\f$m^{-3}\f$]
     */
    virtual double Nc(double T, char point) const;

    /**
     * Get effective density of states in the valance band Nv [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return effective density of states in the valance band Nv [\f$m^{-3}\f$]
     */
    virtual double Nc(double T) const;

    /**
     * Get intrinsic carrier concentration Ni [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return intrinsic carrier concentration Ni [\f$m^{-3}\f$]
     */
    virtual double Ni(double T) const;

    /**
     * Get free carrier concentration N [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return free carrier concentration N [\f$m^{-3}\f$]
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
     * Get mobility [m^2/(V*s)].
     * @param T temperature [K]
     * @return mobility [m^2/(V*s)]
     */
    virtual double mob(double T) const;

    /**
     * Get electrical conductivity Sigma [S/m].
     * @param T temperature [K]
     * @return electrical conductivity Sigma [S/m]
     */
    virtual double cond(double T) const;

    /**
     * Get electrical conductivity in vertical direction Sigma [S/m].
     * @param T temperature [K]
     * @return electrical conductivity in vertical direction Sigma [S/m]
     */
    virtual double cond_v(double T) const;

    /**
     * Get electrical conductivity in lateral direction Sigma [S/m].
     * @param T temperature [K]
     * @return electrical conductivity in lateral direction Sigma[S/m]
     */
    virtual double cond_l(double T) const;

    /**
     * Get electrical resistivity [Ohm*m].
     * @param T temperature [K]
     * @return electrical resistivity [Ohm*m]
     */
    virtual double res(double T) const;

    /**
     * Get electrical resistivity in vertical direction [Ohm*m].
     * @param T temperature [K]
     * @return electrical resistivity in vertical direction [Ohm*m]
     */
    virtual double res_v(double T) const;

    /**
     * Get electrical resistivity in lateral direction [Ohm*m].
     * @param T temperature [K]
     * @return electrical resistivity in vertical direction [Ohm*m]
     */
    virtual double res_l(double T) const;

    /**
     * Get monomolecular recombination coefficient A [1/s].
     * @param T temperature [K]
     * @return monomolecular recombination coefficient A [1/s]
     */
    virtual double A(double T) const;

    /**
     * Get radiative recombination coefficient B[m^3/s].
     * @param T temperature [K]
     * @return radiative recombination coefficient B[m^3/s]
     */
    virtual double B(double T) const;

    /**
     * Get Auger recombination coefficient C [m^6/s].
     * @param T temperature [K]
     * @return Auger recombination coefficient C [m^6/s]
     */
    virtual double C(double T) const;

    /**
     * Get ambipolar diffusion coefficient D[m^2/s].
     * @param T temperature [K]
     * @return ambipolar diffusion coefficient D[m^2/s]
     */
    virtual double D(double T) const;

    /**
     * Get thermal conductivity k[W/(m*K)].
     * @param T temperature [K]
     * @return ambipolar diffusion coefficient D[m^2/s]
     */
    virtual double condT(double T) const;

    /**
     * Get thermal conductivity k[W/(m*K)].
     * @param T temperature [K]
     * @param thickness thickness [m]
     * @return ambipolar diffusion coefficient D[m^2/s]
     */
    virtual double condT(double T, double thickness) const;

    /**
     * Get thermal conductivity in vertical direction k [W/(m*K)].
     * @param T temperature [K]
     * @param thickness thickness [m]
     * @return thermal conductivity in vertical direction k [W/(m*K)]
     */
    virtual double condT_v(double T, double thickness) const;

    /**
     * Get thermal conductivity in lateral direction k [W/(m*K)].
     * @param T temperature [K]
     * @param thickness thickness [m]
     * @return thermal conductivity in lateral direction k [W/(m*K)]
     */
    virtual double condT_l(double T, double thickness) const;

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
    virtual double specHeat(double T) const;

    /**
     * Get refractive index nR [-].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index nR [-]
     */
    virtual double nr(double wl, double T) const;

    /**
     * Get absorption coefficient alpha[\f$cm^{-1}\f$].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @return absorption coefficient alpha[\f$cm^{-1}\f$]
     */
    virtual double absp(double wl, double T) const;

    /**
     * Get refractive index nR[-].
     * @param wl Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index nR[-]
     */
    virtual dcomplex Nr(double wl, double T) const;

    //virtual std::tuple<double, double, double, double, double> Nr(double wl, double T..??..) const;// refractive index (tensor) nR[-]: Wavelength[nm], Temperature[K]

protected:
    void throwNotImplemented(const std::string& method_name) const;
    void throwNotApplicable(const std::string& method_name) const;

};


/**
 * Base maerial class for all semiconductors and similar materials
 */
struct Semiconductor: public Material {
    virtual Kind kind() const;
};

/**
 * Base maerial class for all metals
 */
struct Metal: public Material {
    virtual Kind kind() const;
};

/**
 * Base maerial class for all oxides
 */
struct Oxide: public Material {
    virtual Kind kind() const;
};

/**
 * Base maerial class for all liquid crystals
 */
struct LiquidCrystal: public Material {
    virtual Kind kind() const;
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
 * double avg_VBO = m.VBO(300);
 * @endcode
 */
struct MixedMaterial: public Material {

    /** Vector of materials and its weights. */
    std::vector < std::tuple <shared_ptr<Material>, double> > materials;

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

    //Material methods implementation:
    virtual std::string name() const;

    virtual Kind kind() const;

    virtual double lattC(double T, char x) const;

    virtual double Eg(double T, char point) const;

    virtual double CBO(double T, char point) const;

    virtual double VBO(double T) const;

    virtual double Dso(double T) const;

    virtual double Mso(double T) const;

    virtual double Me(double T, char point) const;

    virtual double Me_v(double T, char point) const;

    virtual double Me_l(double T, char point) const;

    virtual double Mhh(double T, char point) const;

    virtual double Mhh_v(double T, char point) const;

    virtual double Mhh_l(double T, char point) const;

    virtual double Mlh(double T, char point) const;

    virtual double Mlh_v(double T, char point) const;

    virtual double Mlh_l(double T, char point) const;

    virtual double Mh(double T, char EqType) const;

    virtual double Mh_v(double T, char point) const;

    virtual double Mh_l(double T, char point) const;

    virtual double eps(double T) const;

    virtual double chi(double T, char point) const;

    virtual double chi(char point) const;

    virtual double Nc(double T, char point) const;

    virtual double Nc(double T) const;

    virtual double Ni(double T) const;

    virtual double Nf(double T) const;

    virtual double EactD(double T) const;

    virtual double EactA(double T) const;

    virtual double mob(double T) const;

    virtual double cond(double T) const;

    virtual double cond_v(double T) const;

    virtual double cond_l(double T) const;

    virtual double res(double T) const;

    virtual double res_v(double T) const;

    virtual double res_l(double T) const;

    virtual double A(double T) const;

    virtual double B(double T) const;

    virtual double C(double T) const;

    virtual double D(double T) const;

    virtual double condT(double T) const;

    virtual double condT(double T, double thickness) const;

    virtual double condT_v(double T, double thickness) const;

    virtual double condT_l(double T, double thickness) const;

    virtual double dens(double T) const;

    virtual double specHeat(double T) const;

    virtual double nr(double wl, double T) const;

    virtual double absp(double wl, double T) const;

    virtual dcomplex Nr(double wl, double T) const;

private:

    /**
     * Calulate weighted sums of materials (from materials vector) properties.
     * @param f functore which calculate property value for given material
     * @return calculated sum, with the same type which return functor
     * @tparam Functor type of functor wchich can take const Material& argument, and return something which can be multiple by scallar, added, and assigned
     */
    template <typename Functor>
    auto avg(Functor f) const -> typename std::remove_cv<decltype(f(*((const Material*)0)))>::type {
        typename std::remove_cv<decltype(f(*((const Material*)0)))>::type w_sum;
        for (auto& p: materials) {
            w_sum += std::get<1>(p) * f(*std::get<0>(p));
        }
        return w_sum;
    }

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
struct RotatedMaterial: public Material {

    shared_ptr<Material> wrapped;

};

} // namespace plask

#endif	//PLASK__MATERIAL_H
