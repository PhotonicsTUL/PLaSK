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

#define RETURN_MATERIAL_NAN(param) \
    static bool warn = true; \
    if (warn) { writelog(LOG_WARNING, "Material {}: non-applicable parameter " BOOST_PP_STRINGIZE(param) " returned as NAN", name()); warn = false; } \
    return NAN;

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
        CARRIERS_CONCENTRATION   ///< carrier concentration
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
     * Check if dopant is included in @p material_name.
     * @param material_name full material name
     * @return @c true only if dopant is included in @p material_name.
     */
    static bool isNameWithDopant(const std::string& material_name) { return material_name.find(':') != std::string::npos; }



    /**
     * Check if @p material_name is name of simple material.
     * @param material_name full material name or name without dopant (only part before ':')
     * @return @c true only if @p material_name is name of simple material (does not have composition).
     */
    static bool isSimpleMaterialName(const std::string &material_name) { return material_name.find('(') == std::string::npos; }

    /**
     * Parameters of material, information about: name, label, composition and dopant.
     *
     * It stores all information which are represented by material string,
     * but without precision lossing (amounts are stored in doubles).
     *
     * Can be obtained either from string (see parse(std::string)) or material (see getParameters()).
     */
    struct PLASK_API Parameters {

        /// name of material, part before label, can be undefined for complex materials
        std::string name;

        /// material label
        std::string label;

        /// material composition
        Composition composition;

        /// name of dopant
        std::string dopantName;

        /// ammount of dopant (0.0 if there is no dopant)
        double dopingAmount;

        /// type of dopant
        Material::DopingAmountType dopingAmountType;

        /// Construct empty parameters info.
        Parameters(): dopingAmount(0.0), dopingAmountType(NO_DOPING) {}

        /**
         * Construct parameters filled with information parsed from format name[_label][:dopant].
         *
         * Part before label is always put in name, also for complex materials.
         * @param full_material_str material in format name[_label][:dopant]
         * @param allow_dopant_without_amount if true, dopant part without ammount is allowed (in such case, dopantName is filled, but dopingAmountType is set to NO_DOPING and dopingAmount to 0.0)
         */
        explicit Parameters(const std::string& full_name, bool allow_dopant_without_amount = false)
            { parse(full_name, allow_dopant_without_amount); }

        /**
         * Check if material is simple, i.e. has empty composition.
         * @return true only if material is simple
         */
        bool isSimple() const { return composition.empty(); }

        /**
         * Check if dopant name is known.
         * @return true only if dopant name is known (dopingAmountType still can be equal NO_DOPING if name of dopant was given without ammount)
         */
        bool hasDopantName() const { return !dopantName.empty(); }

        /**
         * Check if has full dopant information (with ammount).
         * @return true if has full dopant information
         */
        bool hasDoping() const { return dopingAmountType != NO_DOPING; }

        /**
         * Parse material in format name[_label][:dopant].
         *
         * Part before label is always put in name, also for complex materials.
         * @param full_material_str material in format name[_label][:dopant]
         * @param allow_dopant_without_amount if true, dopant part without ammount is allowed (in such case, dopantName is filled, but dopingAmountType is set to NO_DOPING and dopingAmount to 0.0)
         */
        void parse(const std::string& full_material_str, bool allow_dopant_without_amount = false);

        /**
         * Get complete composition.
         * @return complete composition (without NaNs)
         */
        Composition completeComposition() const;

        /**
         * Set doping parameters.
         * @param dopantName, dopingAmountType, dopingAmount new dopant parameters
         */
        void setDoping(const std::string& dopantName, Material::DopingAmountType dopingAmountType, double dopingAmount);

        /**
         * Clear doping parameters.
         */
        void clearDoping() {
            setDoping("", Material::NO_DOPING, 0.0);
        }
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
     * @param[in] allow_dopant_without_amount if true, dopant without ammount is allowed (in such case, dopant_elem_name is filled, but doping_amount_type is set to NO_DOPING and doping_amount to 0.0)
     */
    static void parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount, bool allow_dopant_without_amount = false);

    /**
     * Parse information about dopant from string.
     *
     * Throws exception in case of parsing errors.
     * @param[in] dopant string to parse
     * @param[out] dopant_elem_name, doping_amount_type, doping_amount parsed values
     * @param[in] allow_dopant_without_amount if true, dopant without ammount is allowed (in such case, dopant_elem_name is filled, but doping_amount_type is set to NO_DOPING and doping_amount to 0.0)
     */
    static void parseDopant(const std::string& dopant, std::string& dopant_elem_name, DopingAmountType& doping_amount_type, double& doping_amount, bool allow_dopant_without_amount = false);

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

    /**
     * Create OpenMP lock guard.
     */
    virtual OmpLockGuard<OmpNestLock> lock() const {
        return OmpLockGuard<OmpNestLock>();
    }

    /// Do nothing.
    virtual ~Material() {}

    /**
     * Get short (without composition and doping amounts) name of material.
     * @return material name
     */
    virtual std::string name() const = 0;

    /**
     * Get dopant material name (part of name after ':', possibly empty).
     * @return dopant material name
     */
    std::string dopantName() const;

    /**
     * Get material name without dopant (without ':' and part of name after it).
     * @return material name without dopant
     */
    std::string nameWithoutDopant() const;

    /**
     * Get full (with composition and doping amounts) name of material.
     *
     * Default implementation returns name, which is fine only for simple materials.
     * @return material name with information about composition and doping
     * @see NameBuilder
     */
    virtual std::string str() const;

    /**
     * Check if @c this material is simple.
     * @return @c true only if @c this material is simple
     */
    bool isSimple() const;

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
    virtual double Eg(double T, double e=0., char point='*') const;

    /**
     * Get conduction band level CB [eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in the Brillouin zone [-]
     * @return conduction band level CB [eV]
     */
    virtual double CB(double T, double e=0., char point='*') const;

    /**
     * Get valence band level VB[eV].
     * @param T temperature [K]
     * @param e lateral strain [-]
     * @param point point in Brillouin zone [-]
     * @param hole hole type ('H'eavy or 'L'ight) [-]
     * @return valence band level VB[eV]
     */
    virtual double VB(double T, double e=0., char point='*', char hole='H') const;

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
    virtual Tensor2<double> Me(double T, double e=0., char point='*') const;

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
	* Get Luttinger parameter y1 [-].
	* @return Luttinger parameter y1 [-]
	*/
	virtual double y1() const;

	/**
	* Get Luttinger parameter y2 [-].
	* @return Luttinger parameter y2 [-]
	*/
	virtual double y2() const;

	/**
	* Get Luttinger parameter y3 [-].
	* @return Luttinger parameter y3 [-]
	*/
	virtual double y3() const;

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
    virtual double chi(double T, double e=0., char point='*') const;

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
     * Get mobility in-plane (lateral) and cross-plane (vertical) direction [cm^2/(V*s)].
     * @param T temperature [K]
     * @return mobility [cm^2/(V*s)]
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
     * Get radiative recombination coefficient B [cm^3/s].
     * @param T temperature [K]
     * @return radiative recombination coefficient B [cm^3/s]
     */
    virtual double B(double T) const;

    /**
     * Get Auger recombination coefficient C [cm^6/s].
     * @param T temperature [K]
     * @return Auger recombination coefficient C [cm^6/s]
     */
    virtual double C(double T) const;

    /**
     * Get ambipolar diffusion coefficient D [cm^2/s].
     * @param T temperature [K]
     * @return ambipolar diffusion coefficient D [cm^2/s]
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
     * @param lam Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return refractive index Nr [-]
     */
    virtual double nr(double lam, double T, double n = 0) const;

    /**
     * Get absorption coefficient alpha [cm^(-1)].
     * @param lam Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return absorption coefficient alpha cm^(-1)]
     */
    virtual double absp(double lam, double T) const;

    /**
     * Get refractive index Nr [-].
     * @param lam Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index Nr[-]
     */
    virtual dcomplex Nr(double lam, double T, double n = 0) const;

    /**
     * Get anisotropic refractive index tensor NR [-].
     * Tensor must have the form \f$ \left[\begin{array}{ccc} n_{0} & n_{3} & 0\\ n_{4} & n_{1} & 0\\ 0 & 0 & n_{2} \end{array}\right] \f$,
     * where \f$ n_i \f$ is i-th object of the returned tuple.
     * @param lam Wavelength [nm]
     * @param T temperature [K]
     * @param n injected carriers concentration [1/cm]
     * @return refractive index tensor NR[-]
     */
    virtual Tensor3<dcomplex> NR(double lam, double T, double n = 0) const;

    // #330:

    /**
     * Get electron mobility in-plane (lateral) and cross-plane (vertical) direction [cm^2/(V*s)].
     * \param T temperature [K]
     * \return mobility [cm^2/(V*s)]
     */
    virtual Tensor2<double> mobe(double T) const;

    /**
     * Get hole mobility in-plane (lateral) and cross-plane (vertical) direction [cm^2/(V*s)].
     * \param T temperature [K]
     * \return mobility [cm^2/(V*s)]
     */
    virtual Tensor2<double> mobh(double T) const;

    /**
     * Get monomolecular electrons lifetime [ns].
     * \param T temperature [K]
     * \return monomolecular electrons lifetime [ns]
     */
    virtual double taue(double T) const;

    /**
     * Get monomolecular holes lifetime [ns].
     * \param T temperature [K]
     * \return monomolecular holes lifetime [ns]
     */
    virtual double tauh(double T) const;

    /**
     * Get Auger recombination coefficient C for electrons [cm^6/s].
     * \param T temperature [K]
     * \return Auger recombination coefficient C [cm^6/s]
     */
    virtual double Ce(double T) const;

    /**
     * Get Auger recombination coefficient C for holes [cm^6/s].
     * \param T temperature [K]
     * \return Auger recombination coefficient C [cm^6/s]
     */
    virtual double Ch(double T) const;

    /**
     * Get piezoelectric constant e13 [C/m^2]
     * \param T temperature [K]
     * \return piezoelectric constant e13 [C/m^2]
     */
    virtual double e13(double T) const;

    /**
     * Get piezoelectric constant e13 [C/m^2]
     * \param T temperature [K]
     * \return piezoelectric constant e13 [C/m^2]
     */
    virtual double e15(double T) const;

    /**
     * Get piezoelectric constant e33 [C/m^2]
     * \param T temperature [K]
     * \return piezoelectric constant e33 [C/m^2]
     */
    virtual double e33(double T) const;

    /**
     * Get elastic constant c13 [GPa].
     * \param T temperature [K]
     * \return elastic constant c13 [GPa]
     */
    virtual double c13(double T) const;

    /**
     * Get elastic constant c33 [GPa].
     * \param T temperature [K]
     * \return elastic constant c33 [GPa]
     */
    virtual double c33(double T) const;

    /**
     * Get spontaneous polarization [C/m^2]
     * \param T temperature [K]
     * \return spontaneous polarization [C/m^2]
     */
    virtual double Psp(double T) const;

    /**
     * Get acceptor concentration Na [cm^(-3)].
     * @return acceptor concentration Na [cm^(-3)]
     */
    virtual double Na() const;

    /**
     * Get donor concentration Nd [cm^(-3)].
     * @return donor concentration Nd [cm^(-3)]
     */
    virtual double Nd() const;



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
    [[noreturn]] void throwNotImplemented(const std::string& method_name) const;
};


/**
 * Dummy named material.
 */
struct PLASK_API DummyMaterial: public Material {

  protected:
    std::string _name;

  public:

    DummyMaterial(const std::string name): _name(name) {}

    std::string name() const override { return _name; }

    Kind kind() const override { return NONE; }
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
 * Empty material, which can actually be instantiated
 */
struct PLASK_API EmptyMaterial : public Material {
    virtual std::string name() const override { return ""; }
    virtual Material::Kind kind() const override { return Material::NONE; }
    virtual bool isEqual(const Material&) const override { return true; } // all empty materials are always equal
};

} // namespace plask

#endif	//PLASK__MATERIAL_H
