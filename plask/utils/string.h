#ifndef PLASK__STRING_H
#define PLASK__STRING_H

/** @file
This file includes string and parsers utils.
*/

#include <string>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

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
 * Calculate copy of string @p str without some characters.
 * @param str string to filter
 * @param pred predictad which return @c true for chars which should stay, and @c false for char which should be removed
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
 * Calculate copy of string @p str with some characters replaced by other.
 * @param str string
 * @param repl functor which return new character or string for each inpu character
 * @return copy of @a str witch replaced all characters by results of @p repl
 * @tparam CharReplacer functor which take one argument (char) and return char or string
 */
template <typename CharReplacer>
std::string replaceChars(const std::string& str, CharReplacer repl) {
    std::string result;
    result.reserve(str.size());
    for (auto c: str) result += repl(c);
    return result;
}

/**
 * @param str string to filter
 * @param chars_to_remove set of characters to remove
 * @return copy of @a str with removed chars which are include in @a chars_to_remove
 */
std::string removedChars(const std::string& str, const std::string& chars_to_remove);

/**
 * Split @p input to sequence of tokens. White spaces on beginning and ending of each token are removed.
 * @param input input string
 * @param pred a predicate to identify separators, this predicate is supposed to return @c true only if a given element is a separator
 * @param eCompress if it is set to boost::algorithm::token_compress_on, adjacent separators are merged together, otherwise, every two separators delimit a token.
 * @return sequence of token
 */
template<typename RangeT, typename PredicateT, typename SequenceSequenceT = std::vector<std::string> >
SequenceSequenceT splitAndTrimPred(RangeT & input, PredicateT pred, boost::algorithm::token_compress_mode_type eCompress = boost::algorithm::token_compress_off) {
    SequenceSequenceT result;
    boost::algorithm::split(result, input, pred, eCompress);
    for (auto& r: result) boost::algorithm::trim(r);
    return result;
}

/**
 * Split @p str to sequence of tokens. White spaces on beginning and ending of each token are removed.
 * @param input input string
 * @param splitter character which separate tokents, typically ',' or ';'
 * @param esc_char escape character which alow to insert separate character in token
 * @param quote_char the character to use for the quote also alow to insert separate character in token
 * @return sequence of token
 */
std::vector<std::string> splitAndTrimEsc(const std::string& str, char splitter, char esc_char = '\\', char quote_char = '\'');

}       // namespace plask

#endif // STRING_H
