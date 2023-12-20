/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__STRING_H
#define PLASK__STRING_H

/** @file
This file contains string and parsers utils.
*/

#include <string>
#include <tuple>
#include <vector>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/tokenizer.hpp>
#include <boost/units/detail/utility.hpp>

#include <plask/config.hpp>   //for PLASK_API

namespace plask {

/**
 * Base class / helper for printable classes with virtual print method.
 */
struct PLASK_API Printable {

    /**
     * Print this to stream @p out.
     * @param out print destination, output stream
     */
    virtual void print(std::ostream& out) const = 0;

    virtual ~Printable();

    /**
     * Print this to stream using print method.
     * @param out print destination, output stream
     * @param to_print vector to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Printable& to_print) {
        to_print.print(out);
        return out;
    }

    /**
     * Get string representation of this using print method.
     * @return string representation of this
     */
    std::string str() const;
};

/**
 * Print all values from sequence [begin, end) sepparating by @p sep.
 * @param out print destination
 * @param begin, end sequence to print
 * @param sep
 * @return out
 */
template <typename Iter>
std::ostream& print_seq(std::ostream& out, Iter begin, Iter end, const char* sep = ", ") {
    if (begin == end) return out;
    out << str(*begin);
    while (++begin != end) { out << sep << str(*begin); }
    return out;
}

/**
 * Split string to two parts: before @a spliter and after @a spliter.
 * If @a spliter is not included in string return pair: @a to_split and empty string.
 * @param to_split string to split
 * @param splitter splitter character
 * @return two strings, @a to_split after split
 */
PLASK_API std::pair<std::string, std::string> splitString2(const std::string& to_split, char splitter);

/**
 * Calculate copy of string @p str without some characters.
 * @param str string to filter
 * @param pred predictad which return @c true for chars which should stay, and @c false for char which should be removed
 * @return copy of @a str which contains only chars for which Pred is @c true
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
 * @param repl functor which return new character or string for each input character
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
PLASK_API std::string removedChars(const std::string& str, const std::string& chars_to_remove);

/**
 * Split @p input to sequence of tokens. White spaces on beginning and ending of each token are removed.
 * @param input input string
 * @param pred a predicate to identify separators, this predicate is supposed to return @c true only if a given object is a separator
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

typedef boost::tokenizer< boost::escaped_list_separator<char> > split_esc_tokenizer;

/**
 * Split @p str to sequence of tokens. White spaces on beginning and ending of each token are removed.
 * @param str input string
 * @param splitter character which separate tokents, typically ',' or ';'
 * @param quote_char the character to use for the quote, allow to insert separate character in token (if token is quoted)
 * @param esc_char escape character which allow to insert separate character in token, typically '\\'
 * @return tokenizer which allow to iterate over tokens
 */
PLASK_API split_esc_tokenizer splitEscIterator(const std::string& str, char splitter, char quote_char = '\'', char esc_char = '\\');

/**
 * Split @p str to sequence of tokens. White spaces on beginning and ending of each token are removed.
 * @param str input string
 * @param splitter character which separate tokents, typically ',' or ';'
 * @param quote_char the character to use for the quote also allow to insert separate character in token
 * @param esc_char escape character which allow to insert separate character in token
 * @return sequence of token
 */
PLASK_API std::vector<std::string> splitEsc(const std::string& str, char splitter, char quote_char = '\'', char esc_char = '\\');

/**
 * Check if @p potential_id is valid C/C++/python name.
 * @param potential_id
 * @return @c true only if @p potential_id is valid C/C++/python name
 */
PLASK_API bool isCid(const char* potential_id);


/// Get simplified type name of given type
template <typename T>
std::string type_name() {
    std::string demangled = boost::units::detail::demangle(typeid(T).name());
    size_t s = demangled.rfind(':');
    if (s == std::string::npos) s = 0; else ++s;
    return demangled.substr(s, demangled.find('<')-s);
}

}       // namespace plask

#endif // STRING_H
