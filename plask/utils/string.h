#ifndef PLASK__STRING_H
#define PLASK__STRING_H

/** @file
This file includes string and parsers utils.
*/

#include <string>
#include <tuple>

namespace plask {

/**
 * Split string to two parts: before @a spliter and after @a spliter.
 * If @a spliter is not included in string return pair: @a to_split and empty string.
 * @param to_split string to split
 * @param spliter spliter character
 * @return two strings, @a to_split after split
 */
std::tuple<std::string, std::string> splitString2(const std::string& to_split, char spliter);

}       // namespace plask

#endif // STRING_H
