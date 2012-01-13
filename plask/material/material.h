#ifndef PLASK__MATERIAL_H
#define PLASK__MATERIAL_H

/** @file
This file includes base classes for materials and material database class.
*/

#include <string>
#include <config.h>
#include <map>
#include <vector>
#include "../exceptions.h"

namespace plask {

/**
 * Represent material, its physical properties.
 */
struct Material {

    /// Do nothing.
    virtual ~Material() {}

    /// @return material name
    virtual std::string name() const = 0;

#ifdef DISABLE___

    /**
     * Get lattice constant [A].
     * @param T temperature [K]
     * @param x lattice paramter [-]
     * @return lattice constant [A]
     */
    virtual double lattC(double T, char x) const;

    /**
     * @param T temperature [K]
     * @param Point point in the Brillouin zone [-]
     * @return energy gap Eg [eV]
     */
    virtual double Eg(double T, char Point) const;

    /**
     * Get conduction band offset CBO [eV].
     * @param T temperature [K]
     * @param Point point in the Brillouin zone [-]
     * @return conduction band offset CBO [eV]
     */
    virtual double CBO(double T, char Point) const;

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
     * @param Point point in Brillouin zone [-]
     * @return split-off mass Mso [\f$m_0\f$]
     */
    virtual double Me(double T, char Point) const;

    /**
     * Get electron effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return electron effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Me_v(double T, char Point) const;

    /**
     * Get electron effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return electron effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Me_l(double T, char Point) const;

    /**
     * Get heavy hole effective mass Mhh [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return heavy hole effective mass Mhh [\f$m_0\f$]
     */
    virtual double Mhh(double T, char Point) const;

    /**
     * Get heavy hole effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return heavy hole effective mass Mhh [\f$m_0\f$]
     */
    virtual double Mhh_v(double T, char Point) const;

    /**
     * Get heavy hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return heavy hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mhh_l(double T, char Point) const;

    /**
     * Get light hole effective mass Mlh [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return light hole effective mass Mlh [\f$m_0\f$]
     */
    virtual double Mlh(double T, char Point) const;

    /**
     * Get light hole effective mass Me in vertical direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return light hole effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Mlh_v(double T, char Point) const;

    /**
     * Get light hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return light hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mlh_l(double T, char Point) const;

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
     * @param Point point in Brillouin zone [-]
     * @return hole effective mass Me in vertical direction [\f$m_0\f$]
     */
    virtual double Mh_v(double T, char Point) const;

    /**
     * Get hole effective mass Me in lateral direction [\f$m_0\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return hole effective mass Me in lateral direction [\f$m_0\f$]
     */
    virtual double Mh_l(double T, char Point) const;

    /**
     * Get dielectric constant EpsR [-].
     * @param T temperature [K]
     * @return dielectric constant EpsR [-]
     */
    virtual double eps(double T) const;

    /**
     * Get electron affinity Chi[eV].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return electron affinity Chi [eV]
     */
    virtual double chi(double T, char Point) const;

    virtual inline double chi(char Point) const {
	return chi(300, Point);
    }

    /**
     * Get effective density of states in the conduction band Nc [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @param Point point in Brillouin zone [-]
     * @return effective density of states in the conduction band Nc [\f$m^{-3}\f$]
     */
    virtual double Nc(double T, char Point) const;

    /**
     * Get effective density of states in the valance band Nv [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return effective density of states in the valance band Nv [\f$m^{-3}\f$]
     */
    virtual double Nc(double T)	const;

    /**
     * Get intrinsic carrier concentration Ni [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return intrinsic carrier concentration Ni [\f$m^{-3}\f$]
     */
    virtual double Ni(double T)	const;

    /**
     * Get free carrier concentration N [\f$m^{-3}\f$].
     * @param T temperature [K]
     * @return free carrier concentration N [\f$m^{-3}\f$]
     */
    virtual double Nf(double T)	const;

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
     * @param Thickness thickness [m]
     * @return ambipolar diffusion coefficient D[m^2/s]
     */
    virtual double thermCond(double T, double Thick) const;

    /**
     * Get thermal conductivity in vertical direction k [W/(m*K)].
     * @param T temperature [K]
     * @param Thickness thickness [m]
     * @return thermal conductivity in vertical direction k [W/(m*K)]
     */
    virtual double thermCond_v(double T, double Thick) const;

    /**
     * Get thermal conductivity in lateral direction k [W/(m*K)].
     * @param T temperature [K]
     * @param Thickness thickness [m]
     * @return thermal conductivity in lateral direction k [W/(m*K)]
     */
    virtual double thermCond_l(double T, double Thick) const;

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
     * @param WaveLen Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index nR [-]
     */
    virtual double nr(double WaveLen, double T) const;

    /**
     * Get absorption coefficient alpha[-].
     * @param WaveLen Wavelength [nm]
     * @param T temperature [K]
     * @return absorption coefficient alpha[-]
     */
    virtual double absp(double WaveLen, double T) const;

    /**
     * Get refractive index nR[-].
     * @param WaveLen Wavelength [nm]
     * @param T temperature [K]
     * @return refractive index nR[-]
     */
    virtual dcomplex Nr(double WaveLen, double T) const;

#endif
    //virtual std::tuple<double, double, double, double, double> Nr(double WaveLen, double T..??..) const;// refractive index (tensor) nR[-]: Wavelength[nm], Temperature[K]

};

/**
 * Material which consist with few real materials.
 * It calculate averages for all properties.
 */
struct MixedMaterial: public Material {

    //std::map<shared_ptr<Material>, double> materials;

};

/**
 * Material which wrap one material and rotate its tensors properties.
 */
struct RotatedMaterial: public Material {

    shared_ptr<Material> wrapped;

};

/**
 * Check if material composition is compatible with pattern and change NaN-s in composition to calculated amounts.
 * @param composition ammounts of elements composition with NaN on position for which amounts has not been taken
 * @param pattern sizes of elements groups
 * @return version of @a composition complement with calculated amounts
 */
std::vector<double> fillMaterialCompositionAmounts(const std::vector<double>& composition, unsigned pattern);

/**
 * Materials database.
 *
 * Create materials with given name, composition and dopand.
 *
 */
struct MaterialsDB {

    ///Amounts of dopands.
    enum DOPANT_AMOUNT_TYPE {
        NO_DOPING,              ///< no dopand
        DOPING_CONCENTRATION,   ///< doping concentration
        CARRIER_CONCENTRATION   ///< carrier concentration
    };

    /**
     * Type of function which construct material.
     * @param name plain material name
     * @param composition amounts of elements, with NaN for each element for composition was not written
     * @param dopant_amount_type type of amount of dopand, needed to interpretation of @a dopant_amount
     * @param dopant_amount amount of dopand, is ignored if @a dopant_amount_type is @c NO_DOPANT
     */
    typedef shared_ptr<Material> construct_material_f(const std::string& name, const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount);

    /**
     * Template of function which construct material with given type.
     * @param composition amounts of elements, with NaN for each element for composition was not writen
     * @param dopant_amount_type type of amount of dopand, needed to interpretation of @a dopant_amount
     * @param dopant_amount amount of dopand, is ignored if @a dopant_amount_type is @c NO_DOPANT
     * @tparam MaterialType type of material to construct, must fill requirements:
     * - inharited from plask::Material
     * - has public, static unsigned COMPOSITION_PATTERN field which determinates sizes of composition groups (for example: 21 means that there are two groups, first group has size 2 and second has size 1)
     * - must have constructor which takes parameters: std::vector<double> composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount
     * - this constructor can suppose that composition is complete (without NaN)
     */
    //TODO set some by methods? what with materials without dopands?
        template <typename MaterialType> shared_ptr<Material> construct(const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) {
        return new MaterialType( fillMaterialCompositionAmounts(MaterialType::COMPOSITION_PATTERN), dopant_amount_type, dopant_amount );
    }

private:
    ///Map: material name -> materials constructors functions
    std::map<std::string, construct_material_f*> constructors;

public:

    /**
     * Create material object.
     * @param parsed_name_with_donor material name with donor name in format material_name[:donor_name], for example: "AlGaN" or "AlGaN:Mg"
     * @param composition amounts of elements, with NaN for each element for composition was not writen
     * @param dopant_amount_type type of amount of dopand, needed to interpetation of @a dopant_amount
     * @param dopant_amount amount of dopand, is ignored if @a dopant_amount_type is @c NO_DOPANT
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @a parsed_name_with_donor
     */
    shared_ptr<Material> get(const std::string& parsed_name_with_donor, const std::vector<double>& composition, DOPANT_AMOUNT_TYPE dopant_amount_type = NO_DOPING, double dopant_amount = 0.0) const;

    /**
     * Create material object.
     * @param name_with_components elements composition in format element1(amount1)...elementN(amountN), where some amounts are optional for example: "Al(0.7)GaN"
     * @param dopant_descr empty string if there is no doping or description of dopant in format elementname=amount or elementname p/n=amount, for example: "Mg=7e18" or "Mg p=7e18"
     * @return constructed material
     * @throw NoSuchMaterial if database doesn't know material with name @a parsed_name_with_donor
     * @throw MaterialParseException if can't parse @a name_with_components or @a dopant_descr
     */
    shared_ptr<Material> get(const std::string& name_with_components, const std::string& dopant_descr) const;

    /**
     * Create material object.
     * @param full_name material name, with encoded parameters in format composition[:dopant], see @ref get(const std::string& name_with_components, const std::string& dopant_descr)
     * @return material with @a full_name
     * @throw NoSuchMaterial if material with given name not exists
     * @throw MaterialParseException if can't parse @a full_name
     */
    shared_ptr<Material> get(const std::string& full_name) const;

    /**
     * Add material to DB. Replace existing material if there is one already in DB.
     * @param name material name (with donor after ':')
     * @param constructor function which can create material instance
     */
    void add(const std::string& name, construct_material_f* constructor);

    /**
     * Add material with given type to DB.
     * All information about material (like name, composition amount pattern) are read from static MaterialType fields.
     */
    //template <typename MaterialType> add();

    /**
     * Fill database with default materials creators.
     */
    //TODO materials will be created
    //void init();
};

} // namespace plask

#endif	//PLASK__MATERIAL_H
