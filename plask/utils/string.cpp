#include "string.h"

#include <boost/lexical_cast.hpp>

namespace plask {

Printable::~Printable() {
}

std::string Printable::str() const {
    return boost::lexical_cast<std::string>(*this);
}

std::pair<std::string, std::string> splitString2(const std::string& to_split, char splitter) {
    std::string::size_type p = to_split.find(splitter);
    return p == std::string::npos ?
            std::pair<std::string, std::string>(to_split, "") :
            std::pair<std::string, std::string>(to_split.substr(0, p), to_split.substr(p+1));

}

std::string removedChars(const std::string& str, const std::string& chars_to_remove) {
    return filterChars(str, [&chars_to_remove](char c) { return chars_to_remove.find(c) == std::string::npos; });
}

split_esc_tokenizer splitEscIterator(const std::string& str, char splitter, char quote_char, char esc_char) {
    return split_esc_tokenizer(str, boost::escaped_list_separator<char>(esc_char, splitter, quote_char));
}

std::vector<std::string> splitEsc(const std::string& str, char splitter, char quote_char, char esc_char) {
    boost::escaped_list_separator<char> Separator(esc_char, splitter, quote_char);
    boost::tokenizer< boost::escaped_list_separator<char> > tok( str, Separator );
    return std::vector<std::string>(tok.begin(), tok.end());
}

inline bool isEngLower(char ch) {
    return 'a' <= ch && ch <= 'z';
}

inline bool isEngUpper(char ch) {
    return 'A' <= ch && ch <= 'Z';
}

bool isEngAlpha(char ch) { return isEngLower(ch) || isEngUpper(ch); }

bool isDigit(char ch) { return '0' <= ch && ch <= '9'; }

bool isCid(const char* potential_id, char underline_ch) {
    if (!isEngAlpha(*potential_id) && *potential_id != underline_ch)
        return false;   //first should be letter or underline
    for (++potential_id; *potential_id; ++potential_id) //all next, if are non NULL
        if (!isEngAlpha(*potential_id) && !isDigit(*potential_id) && *potential_id != underline_ch)    //must be letter, digit or underline
            return false;
    return true;
}

}       // namespace plask
