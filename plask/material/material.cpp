#include "material.h"

#include "../utils/string.h"
#include <boost/lexical_cast.hpp>

#include <cmath>

namespace plask {

void fillGroupMaterialCompositionAmounts(std::vector<double>::iterator begin, std::vector<double>::iterator end) {
    auto no_info = end;
    double sum = 0.0;
    for (auto i = begin; i != end; ++i) {
        if (isnan(*i)) {
            if (no_info != end)
                throw plask::MaterialParseException("Both elements in group have no information about composition amount");
            else
                no_info = i;
        } else {
            sum += *i;
            if (sum > 1.0)
                throw plask::MaterialParseException("Sum of composition ammounts in group exceeds 1");
        }
    }
    if (no_info != end) {
        *no_info = 1.0 - sum;
    } else {
        if (!is_zero(sum - 1.0))
             throw plask::MaterialParseException("Sum of composition ammounts in group diffrent from 1");
    }
}

void fillMaterialCompositionAmountsI(std::vector< double >& composition, unsigned int pattern) {
    auto end = composition.end();
    while (pattern != 0) {
        unsigned group_size = pattern % 10;     //last group size
        if (end - composition.begin() < group_size)
            throw plask::CriticalException("Wrong material composition pattern");
        auto begin = end - group_size;
        fillGroupMaterialCompositionAmounts(begin, end);
        end = begin;
        pattern /= 10;
    }
    if (end != composition.end())
        throw plask::CriticalException("Wrong material composition pattern");
}

std::vector< double > fillMaterialCompositionAmounts(const std::vector< double >& composition, unsigned int pattern) {
    std::vector<double> result = composition;
    fillMaterialCompositionAmountsI(result, pattern);
    return result;
}


shared_ptr< Material > plask::MaterialsDB::get(const std::string& parsed_name_with_donor, const std::vector< double >& composition,
                                                    DOPANT_AMOUNT_TYPE dopant_amount_type, double dopant_amount) const
{
    auto it = constructors.find(parsed_name_with_donor);
    if (it == constructors.end()) throw NoSuchMaterial(parsed_name_with_donor);
    return shared_ptr<Material>(it->second(composition, dopant_amount_type, dopant_amount));
}

const char* getElementEnd(const char* begin, const char* end) {
    if (!('A' <= *begin && *begin <= 'Z')) return begin;
    do { ++begin; } while (begin != end && 'a' <= *begin && *begin <= 'z');
    return begin;
}

const char* getAmountEnd(const char* begin, const char* end) {
    if (*begin != '(') return begin;
    do { ++begin; } while (begin != end && *begin != ')');
    return begin;
}

double toDouble(const std::string& s) {
    try {
        return boost::lexical_cast<double>(s);
    } catch (std::exception& e) {
        throw MaterialParseException(e.what());
    }
}

void parseNameWithComposition(const char* begin, const char* end, std::vector<std::string>& components, std::vector<double>& components_amounts) {
    while (begin != end) {
        const char* comp_end = getElementEnd(begin, end);
        if (comp_end == begin)
            throw MaterialParseException(std::string("Expected element but found character: ") + *begin);
        components.push_back(std::string(begin, comp_end));
        const char* amount_end = getAmountEnd(comp_end, end);
        if (amount_end == comp_end) {       //no amount info for this element
            components_amounts.push_back(std::numeric_limits<double>::quiet_NaN());
            begin = amount_end;
        } else {
            if (amount_end == end)
                throw MaterialParseException("Unexpected end of input while reading amount of element. Couldn't find ')'");
            components_amounts.push_back(toDouble(std::string(comp_end+1, amount_end)));
            begin = amount_end+1;   //skip also ')', begin now points to 1 character after ')'
        }
    }
}

void parseDopant(const char* begin, const char* end, std::string& dopant_elem_name, MaterialsDB::DOPANT_AMOUNT_TYPE& dopant_amount_type, double& dopant_amount) {
    const char* name_end = getElementEnd(begin, end);
    if (name_end == begin)
         throw MaterialParseException("No dopant name");
    dopant_elem_name.assign(begin, name_end);
    if (*name_end == '=') {
        if (name_end+1 == end) throw MaterialParseException("Unexpected end of input while reading dopants concentation");
        dopant_amount_type = MaterialsDB::DOPING_CONCENTRATION;
        dopant_amount = toDouble(std::string(name_end+1, end));
        return;
    }
    if (!isspace(*name_end))
        throw MaterialParseException("Expected space or '=' but found '%1%' instead", *name_end);
    do {  ++name_end; } while (name_end != end && isspace(*name_end));   //skip whites
    auto p = splitString2(std::string(name_end, end), '=');
    //TODO check std::get<0>(p) if is p/n compatibile with dopant_elem_name
    dopant_amount_type = MaterialsDB::CARRIER_CONCENTRATION;
    dopant_amount = toDouble(std::get<1>(p));
}

shared_ptr< Material > MaterialsDB::get(const std::string& name_with_components, const std::string& dopant_descr) const {

    std::vector<std::string> components;
    std::vector<double> components_amounts;
    parseNameWithComposition(name_with_components.data(), name_with_components.data() + name_with_components.size(), components, components_amounts);

    std::string parsed_name_with_dopant;
    for (std::string& c: components) parsed_name_with_dopant += c;

    double dopant_amount = 0.0;
    DOPANT_AMOUNT_TYPE dopant_amount_type = NO_DOPANT;
    if (!dopant_descr.empty()) {
        std::string dopant_name;
        parseDopant(dopant_descr.data(), dopant_descr.data() + dopant_descr.size(), dopant_name, dopant_amount_type, dopant_amount);
        parsed_name_with_dopant += ':';
        parsed_name_with_dopant += dopant_name;
    }

    return get(parsed_name_with_dopant, components_amounts, dopant_amount_type, dopant_amount);
}

shared_ptr< Material > MaterialsDB::get(const std::string& full_name) const {
    auto pair = splitString2(full_name, ':');
    return get(std::get<0>(pair), std::get<1>(pair));
}

void MaterialsDB::add(const std::string& name, plask::MaterialsDB::construct_material_f* constructor) {
    constructors[name] = constructor;
}


}       // namespace plask
