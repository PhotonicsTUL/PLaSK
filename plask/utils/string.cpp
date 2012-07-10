#include "string.h"

namespace plask {

std::tuple< std::string, std::string > splitString2(const std::string& to_split, char splitter) {
    std::string::size_type p = to_split.find(splitter);
    return p == std::string::npos ?
            std::tuple<std::string, std::string>(to_split, "") :
            std::tuple<std::string, std::string>(to_split.substr(0, p), to_split.substr(p+1));

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

}       // namespace plask
