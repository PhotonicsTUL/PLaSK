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
 * @param splitter splitter character
 * @return two strings, @a to_split after split
 */
std::tuple<std::string, std::string> splitString2(const std::string& to_split, char splitter);

/**
 * @param pred predictad which return @c true for chars which should stay, and @c false for char which should be removed 
 * @param str string to filter
 * @return copy of @a str wich includes only chars for which Pred is @c true
 * @tparam Pred functor which take one argument (char) and return bool
 */
template <typename Pred>
std::string filterChars(const std::string& str, Pred pred) {
    std::string result;
    for (auto c: str) if (pred(c)) result += c;
    return result;
}

/**
 * @param str string to filter
 * @param chars_to_remove set of characters to remove
 * @return copy of @a str with removed chars which are include in @a chars_to_remove
 */
std::string removedChars(const std::string& str, const std::string& chars_to_remove);

}       // namespace plask

#endif // STRING_H
