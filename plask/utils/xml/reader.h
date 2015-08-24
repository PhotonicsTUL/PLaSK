#ifndef PLASK__UTILS_XML_READER_H
#define PLASK__UTILS_XML_READER_H

#include <string>
#include <limits>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <set>
#include <map>

#include <typeinfo>
#include <typeindex>
#include <functional>
#include <type_traits>
#include <boost/any.hpp>

#include "exceptions.h"

//this is copy paste from expat.h, this allows to not include expat.h in header
extern "C" {
    struct XML_ParserStruct;
    typedef struct XML_ParserStruct *XML_Parser;
}

namespace plask {

/**
 * Objects of this class allow to parse string value and interpret it as given type.
 *
 * By default boost::lexical_cast is used to parse, but this can be changed, for each required result type, to custom parser.
 */
class PLASK_API StringInterpreter {

    typedef std::function<boost::any(const std::string&)> type_parser;

    std::map<std::type_index, type_parser> parsers;

    void set() {}   //do nothing, defined for stop of set recursion and default constructor

public:

    /**
     * Construct StringInterpreter which uses given parsers for types returned by this parsers, and boost::lexical_cast for other types.
     * @param parsers 0 or more custom parsers, functors which can take std::string and return value of some type,
     *                  or throw excpetion in case of parsing error
     */
    template <typename... Functors>
    StringInterpreter(Functors... parsers) { set(parsers...); }

    StringInterpreter(StringInterpreter&&) = default;
    StringInterpreter(const StringInterpreter&) = default;
    StringInterpreter& operator=(StringInterpreter&&) = default;
    StringInterpreter& operator=(const StringInterpreter&) = default;

    /**
     * Parse given text @a str, interpret it as type @p RequiredType.
     *
     * For parsing, it uses registred interpreter or boost::lexical_cast.
     *
     * It throws exception in case of parsing error.
     * @param str text to parse
     * @return @p @a str interpreted as type @p RequiredType
     */
    template <typename RequiredType>
    RequiredType get(const std::string& str) const {
        auto i = parsers.find(std::type_index(typeid((RequiredType*)0)));
        if (i != parsers.end())
            return boost::any_cast<RequiredType>(i->second(str));
        return boost::lexical_cast<RequiredType>(boost::trim_copy(str));
    }

    /**
     * Set parsers to use (interpret attributes values, etc.) for conversion from std::string to type returned by each @a parser.
     * @param parser1, rest_parsers 1 or more functors which can take std::string and return value of some type,
     *                  or throw excpetion in case of parsing error
     */
    template <typename Functor1, typename... Functors>
    void set(Functor1 parser1, Functors... rest_parsers) {
        parsers[std::type_index(typeid((typename std::result_of<Functor1(std::string)>::type*)0))] = parser1;
        set(rest_parsers...);
    }

    /**
     * Unset parser to use (interpret attributes values, etc.) for conversion from std::string to given @a type.
     *
     * Default lexical_cast will be used for given @a type after calling this.
     * @tparam type type returned by parser to unregister
     */
    template <typename type>
    void unset() {
        parsers.erase(std::type_index(typeid((type*)0)));
    }

};

/**
 * XML pull parser.
 *
 * It makes some checks while reading and throw exceptions when XML document not is valid:
 * - it check open/close tags,
 * - it checks if all attributes was read.
 */
class PLASK_API XMLReader {
  public:

    /// Enumeration for all xml nodes which are parsed by XMLReader
    enum NodeType {
    //        NODE_NONE = irr::io::EXN_NONE,                  //< No xml node. This is usually the node if you did not read anything yet.
            NODE_ELEMENT = 1 ,             ///< A xml element, like \<foo>
            NODE_ELEMENT_END = 2,          ///< End of an xml element, like \</foo>
            NODE_TEXT = 4                  ///< Text within a xml element: \<foo> this is the text. \</foo>
    //        NODE_COMMENT =  irr::io::EXN_COMMENT,           //< An xml comment like &lt;!-- I am a comment --&gt; or a DTD definition.
    //        NODE_CDATA =  irr::io::EXN_CDATA,               //< An xml cdata section like &lt;![CDATA[ this is some CDATA ]]&gt;
    //        NODE_UNKNOWN = irr::io::EXN_UNKNOWN             //< Unknown element
    };

    /// Base class to source of data.
    struct PLASK_API DataSource {

        /**
         * Read @p buf_size bytes of data and store it in buffer @p buff.
         *
         * Throws exception if can't read.
         * @param buff destination buffer
         * @param buf_size size of @p buff
         * @return number of bytes read, less than @p buf_size only at the end of data
         */
        virtual std::size_t read(char* buff, std::size_t buf_size) = 0;

        /// Empty, virtual destructor.
        virtual ~DataSource() {}
    };

    /// Allows to read XML from standard C++ input stream (std::istream).
    struct PLASK_API StreamDataSource: public DataSource {

        /// Stream to read from.
        std::unique_ptr<std::istream> input;

        /**
         * @param params arguments for std::unique_ptr<std::istream> constructor
         */
        template <typename... Args>
        StreamDataSource(Args&&... params): input(std::forward<Args>(params)...) {}

        std::size_t read(char* buff, std::size_t buf_size);

    };

    /// Allows to read XML from old C FILE.
    struct PLASK_API CFileDataSource: public DataSource {

        /// Stream to read from.
        FILE* desc;

        /**
        * @param desc stream to read from
        */
        CFileDataSource(FILE* desc): desc(desc) {}

        ~CFileDataSource() { fclose(desc); }

        std::size_t read(char* buff, std::size_t buf_size);

    };


    /// Enum attribute reader class
    template <typename EnumT>
    struct EnumAttributeReader {

      protected:

        XMLReader& reader;
        const std::string& attr_name;
        bool case_insensitive;

        std::map<std::string, EnumT> values;
        std::string help;

        EnumT parse(std::string value) {
            if (case_insensitive) boost::to_lower(value);
            auto found = values.find(value);
            if (found == values.end())
                throw XMLBadAttrException(reader, attr_name, value, "one of " + help);
            return found->second;
        }

      public:

        /**
         * Create enum attribute reader
         * \param reader XML reader
         * \param attr_name name of the attribute
         * \param case_sensitive true if the attribute value should be case sensitive
         **/
        EnumAttributeReader(XMLReader& reader, const std::string& attr_name, bool case_sensitive=false):
            reader(reader), attr_name(attr_name), case_insensitive(!case_sensitive) {}

        /**
         * Add allowed parameter
         * \param key text representing the attribute value
         * \param val value of the attribute
         * \param min minimum number of letters in the attribute
         **/
        EnumAttributeReader& value(std::string key, EnumT val, size_t min=std::numeric_limits<std::size_t>::max()) {
            if (case_insensitive) boost::to_lower(key);
#           ifndef NDEBUG
                if (values.find(key) != values.end()) throw XMLException(reader, "CODE ERROR: Attribute value \"" + key + "\" already defined.");
#           endif
            help += values.empty()? "\"" : ", \"";
            values[key] = val;
            if (min < key.length()) {
                std::string skey = key.substr(0, min);
#               ifndef NDEBUG
                    if (values.find(skey) != values.end()) throw  XMLException(reader, "CODE ERROR: Attribute value \"" + skey + "\" already defined.");
#               endif
                values[skey] = val;
                help += skey; help += "["; help += key.substr(min); help += "]";
            } else
                help += key;
            help += "\"";
            return *this;
        }

        /// Require attribute
        EnumT require() {
            return parse(reader.requireAttribute(attr_name));
        }

        /**
         * Get attribute
         * \return optional set if the attribute was specified
         */
        boost::optional<EnumT> get() {
            boost::optional<std::string> value = reader.getAttribute(attr_name);
            if (!value) return boost::optional<EnumT>();
            return parse(*value);
        }

        /**
         * Get attribute with default value
         * \param default_value default value of the attribute
         * \return optional set if the attribute was specified
         */
        EnumT get(EnumT default_value) {
            boost::optional<std::string> value = reader.getAttribute(attr_name);
            if (!value) return default_value;
            return parse(*value);
        }

    };

    /**
     * Util class that allow to check of duplication of tags. Example usage:
     * <code>
     * XMLReader::CheckTagDuplication dub_check;
     * dub_checke(reader);
     * reader.require_next();
     * dub_checke(reader);  //throw if tag has the same name as previouse one
     * </code>
     */
    class PLASK_API CheckTagDuplication {

        std::set<std::string> namesAlreadySeen;

    public:

        /**
         * Throw exception if tag with name @p name has been already seen and so is not allowed again.
         * @param scope scope, where tag duplication is not allowed (used in formatting of error message)
         * @param name name of tag
         */
        void operator()(const std::string& scope, std::string name) {
            if (namesAlreadySeen.find(name) != namesAlreadySeen.end()) throw XMLDuplicatedElementException(scope, "tag <" + name + ">");
            namesAlreadySeen.insert(std::move(name));
        }

        /**
         * Throw exception if tag with name @p name has been already seen and so is not allowed again.
         * @param scope scope, where tag duplication is not allowed (used in formatting of error message)
         * @param name name of tag
         */
        void operator()(const XMLReader& scope, std::string name) {
            if (namesAlreadySeen.find(name) != namesAlreadySeen.end()) throw XMLDuplicatedElementException(scope, "tag <" + name + ">");
            namesAlreadySeen.insert(std::move(name));
        }

        void operator()(const XMLReader& reader) {
            this->operator ()(reader, reader.getNodeName());
        }

    };

    /// Filter can change attribute or content value before parse.
    typedef std::function<std::string(const std::string&)> Filter;

  private:

    static void startTag(void *data, const char *element, const char **attribute);
    static void endTag(void *data, const char *element);
    static void characterData(void* data, const char *string, int string_len);

    /// Source of data.
    std::unique_ptr<DataSource> source;

    template <typename RequiredType>
    RequiredType parse(const std::string& attr_str) const {
        return stringInterpreter.get<RequiredType>(attr_str);
    }

    template <typename RequiredType>
    RequiredType parse(const std::string& attr_str, const std::string& attr_name) const {
        try {
            return parse<RequiredType>(attr_str);
        } catch (...) {
            throw XMLBadAttrException(*this, attr_name, attr_str);
        }
    }

    /**
     * Fragment of XML data which was read by parser.
     */
    struct State {

        /// Line number when fragment begins.
        unsigned lineNr;

        /// Column number in line this->lineNr when fragment begins.
        unsigned columnNr;

        /// Text or tag name.
        std::string text;

        /// Attributes (used only if type == NODE_ELEMENT).
        std::map<std::string, std::string> attributes;

        /// Type of tag.
        NodeType type;

        /**
         * Construct state using given data.
         * @param type type of tag
         * @param lineNr line number when fragment begins
         * @param columnNr column number (in line @p lineNr) when fragment begins
         * @param text text or tag name
         */
        State(NodeType type, unsigned lineNr, unsigned columnNr, const std::string& text): lineNr(lineNr), columnNr(columnNr), text(text), type(type) {}

        /**
         * Check if text consist of white characters only.
         * @return @p true if text consist of white characters only
         */
        bool hasWhiteText() {
            for (std::size_t i = 0; i < text.size(); ++i)
                if (!isspace(text[i])) return false;
            return true;
        }
    };

    /**
     * Append parsed XML fragment.
     * @param type type of tag
     * @param text text or tag name
     */
    State& appendState(NodeType type, const std::string& text);

    /// Parsed states (if last one is NODE_TEXT than in can be not complatly parsed).
    std::deque<State> states;

    /// XML reader/parser, low level.
    XML_Parser parser;

    /// Path from root to the current tag.
    std::vector<std::string> path;

    /// Attributes which was read in the current tag.
    std::set<std::string> read_attributes;

public:
    /// Parsers used to interpret string values.
    StringInterpreter stringInterpreter;

    /// Filter that can change attribute value before parse.
    Filter attributeFilter;

    /// Filter that can change content value before parse.
    Filter contentFilter;

    /**
     * Set attribute and content filter to @p f.
     * @param f new filter common for attribute and content
     */
    void setFilter(Filter f) {
        attributeFilter = f;
        contentFilter = f;
    }

private:

    /// true if the reader should check if there are no spurious attributes in the current element
    bool check_if_all_attributes_were_read;

    /**
     * Check if current XML state is available.
     * @return @c true if current XML state is available and can be use, @c false in other cases (typically before first or after last read())
     */
    bool hasCurrent() const {
        if (states.empty()) return false;
        return states.size() > 1 || states.front().type != NODE_TEXT;
    }

    /**
     * Check if current XML state is available and throw exception if not.
     */
    void ensureHasCurrent() const {
        if (!hasCurrent()) throw XMLException("XML reader: no current node (missing first read() call?)");
    }

    /// @return current XML state (valid only if hasCurrent() returns @c true)
    const State& getCurrent() const {
        return states.front();
    }

    /**
     * Parse @p str as bool.
     * @param str string which represent bool: "0", "1", "yes", "no", "true", "false" (case insensive)
     * @return bool parsed from @p str
     */
    static bool strToBool(std::string str) {
        boost::algorithm::to_lower(str);
        if (str == "yes" || str == "true" || str == "1") return true;
        else if (str == "no" || str == "false" || str == "0") return false;
        else throw XMLException("\"" + str + "\" is not valid bool value.");
    }

    /**
     * Parse some data from input. Can (but not must!) append some states to states deque
     * @return @c true if has more data to read and @c false if end of source was reach while reading
     */
    bool readSome();

    /// Initialize XML parser, called by constructors.
    void initParser();

  public:

    /**
     * Throw exception which include information about current position in XML and typically describe logic error in XML file.
     * @param msg custom part of exception message
     */
    void throwException(const std::string& msg) const { throw XMLException(*this, msg); }

    /**
     * Throw XMLUnexpectedEndException.
     */
    void throwUnexpectedEndException() const { throw XMLUnexpectedEndException(*this); }

    /**
     * Throw XMLUnexpectedElementException.
     * @param args... XMLUnexpectedElementException constructor arguments to use, exluding the first one
     */
    template <typename... Args>
    void throwUnexpectedElementException(Args&&... args) const { throw XMLUnexpectedElementException(*this, std::forward<Args>(args)...); }

    /**
     * Throw XMLUnexpectedElementException if node type is not included in required_types or
     * (only when new_tag_name is given) node type is NODE_ELEMENT and name is not equal to new_tag_name.
     * @param required_types bit sum of NodeType-s
     * @param new_tag_name (optional) name of required tag (ingored if NODE_ELEMENT is not included in required_types)
     * @return type of current node
     */
    NodeType requireNodeType(int required_types, const char* new_tag_name = nullptr) const;

    /**
     * Construct XML reader to read XML from given source.
     * @param source source of XML data, will be delete by this after use by this
     */
    XMLReader(std::unique_ptr<DataSource>&& source);

    /**
     * Construct XML reader to read XML from given stream.
     * @param istream stream to read, will be closed and delete by this
     */
    XMLReader(std::unique_ptr<std::istream>&& istream);

    /**
     * Construct XML reader to read XML from given file.
     * @param file_name name of file to read
     */
    XMLReader(const char* file_name);

    /**
     * Construct XML reader to read XML from given @p file FILE*.
     * @param file input file
     */
    XMLReader(FILE* file);

    /*
     * Construct XML reader to read XML from given @p input stream.
     * @param input input stream
     */
    //XMLReader(std::istream& input);

#if (__cplusplus >= 201103L) || defined(__GXX_EXPERIMENTAL_CXX0X__)
    /**
     * Move constructor.
     * @param to_move object to move from, should not be used but only deleted after move
     */
    XMLReader(XMLReader&& to_move);

    /**
     * Move assigment operator.
     * @param to_move object to move from, should not be used but only deleted after move
     * @return *this
     */
    XMLReader& operator=(XMLReader&& to_move);

    /// Disallow copying of reader.
    XMLReader(const XMLReader& to_copy) = delete;
    XMLReader& operator=(const XMLReader& to_copy) = delete;
#else
  private:
    XMLReader(const XMLReader&) {}
    XMLReader& operator=(const XMLReader&) { return *this; }
  public:
#endif

    /// Delete source and XML parser.
    ~XMLReader();

    /**
     * Swap states of @c this and @p to_swap.
     * @param to_swap object to swap with this
     */
    void swap(XMLReader& to_swap);

    /**
     * Get current type of node.
     * @return current type of node
     */
    NodeType getNodeType() const { ensureHasCurrent(); return getCurrent().type; }

    /**
     * Get line number where current element starts.
     * @return line number of current element start
     */
    unsigned getLineNr() const { ensureHasCurrent(); return getCurrent().lineNr; }

    /**
     * Get column number where current element starts.
     * @return column number of current element start
     */
    unsigned getColumnNr() const { ensureHasCurrent(); return getCurrent().columnNr; }

    /**
     * Reads forward to the next xml node.
     * @return @c false only if there is no further node.
     */
    bool next();

    /*
     * Check if node is empty, like \<foo /\>.
     *
     * Note that empty nodes are comunicate by parser two times: as NODE_ELEMENT and next as NODE_ELEMENT_END.
     * So for \<foo /\> parser work just like for \<foo>\</foo> and only this method allow to check which notation was used.
     * @return if an element is an empty element, like \<foo /\>
     */
    //bool isEmptyElement() const { return irrReader->isEmptyElement(); }

    /**
     * Get vector of names of all opened tags from root to current one.
     * Includes tag which is just closed when the current node is NODE_ELEMENT_END.
     * @return vector of names of all opened tags, first is root, last is current tag
     */
    const std::vector<std::string>& getPath() const { return path; }

    /**
     * Get level of current node. Root has level 1, children of root have level 2, and so on.
     * @return level of current node which is equal to length of path returned by getPath()
     */
    std::size_t getLevel() const { return path.size(); }

    /**
     * Returns attribute count of the current XML node.
     *
     * This is usually non 0 if the current node is NODE_ELEMENT, and the element has attributes.
     * @return amount of attributes of this xml node.
     */
    std::size_t getAttributeCount() const { ensureHasCurrent(); return getCurrent().attributes.size(); }

    /**
     * Get all attributes, empty if current node is not NODE_ELEMENT.
     * @return all attributes, reference is valid up to read() call
     */
    const std::map<std::string, std::string>& getAttributes();

    /**
     * Mark attribute with given name as read, so parser does not throw an exception if this attribute will be not read.
     * @param name name of attribute to ignore
     */
    void ignoreAttribute(const std::string& name) { getAttribute(name); }

    /**
     * Allow to have unread attributes in currect tag.
     */
    void ignoreAllAttributes() { check_if_all_attributes_were_read = false; }

    /**
     * Check if current node has attribute with given @p name.
     * @param name attribute name
     * @return @c true only if current node has attribute with given @p name
     */
    bool hasAttribute(const std::string& name) const { return getAttribute(name); }

    /**
     * Create EnumAttributeReader
     * \param attr_name name of the attribute
     * \param case_sensitive true if the attribute value should be case sensitive
     */
    template <typename EnumT>
    EnumAttributeReader<EnumT> enumAttribute(const std::string attr_name, bool case_sensitive=false) {
        return EnumAttributeReader<EnumT>(*this, attr_name, case_sensitive);
    }

    /**
     * Remove from attributes all attributes which are not in the default (empty) namespace.
     */
    void removeAlienNamespaceAttr();

    /**
     * Returns the name of the current node (tag).
     *
     * Throw exception if it is not defined.
     * @return name of the current node
     */
    std::string getNodeName() const;

    /**
     * Returns the name of the current node (tag).
     *
     * Throw exception if it is not defined.
     * @return name of the current node
     */
    inline std::string getTagName() const { return getNodeName(); }

    /**
     * Check if current node is NODE_TEXT (throw exception if it's not) and get node data (text content).
     * @return data of the current node
     */
    std::string getTextContent() const;

    /**
     * Check if current node is NODE_TEXT (throw excpetion if it's not) and get node data (text content).
     * @return data of the current node casted (by lexical_cast) to given type T
     */
    template <typename T>
    inline T getTextContent() const {
        return parse<T>(getTextContent());
    }

    /**
     * Get value of attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node.
     * @param name name of attribute
     * @param default_value default value which will be return when attribute with given @p name is not defined
     * @return attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node
     * @tparam T required type of value, boost::lexical_cast\<T> or registered parser will be used to obtain value of this type from string
     */
    template <typename T>
    inline T getAttribute(const std::string& name, const T& default_value) const {
        boost::optional<std::string> attr_str = getAttribute(name);
        if (attr_str) {
            return parse<T>(*attr_str, name);
        } else
            return default_value;
    }

    /**
     * Get value of attribute with given @p name.
     * @param name name of attribute to get
     * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
     */
    boost::optional<std::string> getAttribute(const std::string& name) const;

    /**
     * Get value of attribute with given @p name.
     *
     * Throws exception if value of attribute given @p name can't be casted to required type T.
     * @param name name of attribute to get
     * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
     * @tparam T required type of value, boost::lexical_cast\<T> or registered parser will be used to obtain value of this type from string
     */
    template <typename T>
    inline boost::optional<T> getAttribute(const std::string& name) const {
        boost::optional<std::string> attr_str = getAttribute(name);
        if (!attr_str) return boost::optional<T>();
        return parse<T>(*attr_str, name);
    }

    /**
     * Require the attribute with given \p name.
     *
     * Throws exception if there is no attribute with given \p name.
     * \return its value
     */
    std::string requireAttribute(const std::string& attr_name) const;

    /**
     * Require the attribute with given \p name.
     *
     * Throws exception if there is no attribute with given \p name.
     * \return its value
     * \tparam T required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
     */
    template <typename T>
    inline T requireAttribute(const std::string& name) const {
        return parse<T>(requireAttribute(name), name);
    }

    /**
     * Call read(), one or more times skipping comments.
     * @throw XMLUnexpectedEndException if there is no next element
     */
    void requireNext();

    /**
     * Call requireNext() and next check if current element is tag opening. Throw exception if it's not.
     */
    void requireTag();

    /**
     * Call requireNext() and next check if current element is tag opening.
     * Throw exception if it's not or if it name is not \p name.
     */
    void requireTag(const std::string& name);

    /**
     * Call requireNext() and next check if current element is tag opening or closing of tag.
     * Throw exception if it's not.
     * \return true if the next tag was opened
     */
    bool requireTagOrEnd();

    /**
     * Call requireNext() and next check if current element is tag opening (in such case it also check if it has name equal to given @p name) or closing of tag.
     * Throw exception if it's not.
     * @param name required name of opening tag
     * @return true if the next tag was opened
     */
    bool requireTagOrEnd(const std::string &name);

    /**
     * Call requireNext() and next check if current element is tag closing. Throw exception if it's not.
     */
    void requireTagEnd();

    /**
     * Call requireNext() and next check if current element is text. Throw exception if it's not.
     * \return read text
     */
    std::string requireText();

    /**
     * Read text inside current tag. Move parser to end of current tag.
     * \return read text
     */
    std::string requireTextInCurrentTag();

    /**
     * Call requireNext() and next check if current element is text. Throw exception if it's not.
     * \return read text casted (by lexical_cast) to givent type T
     */
    template <typename T>
    inline T requireText() {
        return parse<T>(requireText());
    }

    /**
     * Call requireNext() and next check if current element is text. Throw exception if it's not. Next require end of tag.
     * \return read text casted (by lexical_cast) to givent type T
     */
    template <typename T>
    inline T requireTextInCurrentTag() {
        return parse<T>(requireTextInCurrentTag());
    }

    /**
     * Skip everything up to element with required type on required level.
     * @param required_level level on which required element should be
     * @param required_type type of required element
     * @return @c true if reader is on required element or @c false if XML data end
     */
    bool gotoNextOnLevel(std::size_t required_level, NodeType required_type = NODE_ELEMENT);

    /**
     * Skip everything up to next tag element on current level.
     * @return @c true if reader is on required element or @c false if XML data end
     */
    bool gotoNextTagOnCurrentLevel();

    /**
     * Skip everything up to end of the current tag.
     */
    void gotoEndOfCurrentTag();

};

}   // namespace plask


namespace std {
inline void swap(plask::XMLReader& a, plask::XMLReader& b) { a.swap(b); }
}   // namespace std

#endif // PLASK__UTILS_XML_READER_H
