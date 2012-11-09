#ifndef PLASK__UTILS_XML_READER_H
#define PLASK__UTILS_XML_READER_H

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <unordered_set>
#include <map>

#include "exceptions.h"

//this is copy paste from expat.h, it allow to not include expat.h in header
#ifdef __cplusplus
extern "C" {
#endif
struct XML_ParserStruct;
typedef struct XML_ParserStruct *XML_Parser;
#ifdef __cplusplus
}
#endif

namespace plask {

/**
 * XML pull parser.
 *
 * It makes some checks while reading and throw exeptions when XML document not is valid:
 * - it check open/close tags,
 * - it checks if all attributes was read.
 */
class XMLReader {
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

    struct DataSource {
        /**
         * @brief read
         * @param buff
         * @param buf_size
         * @return number of bytes read, less than @p buf_size only on end of data
         */
        virtual std::size_t read(char* buff, std::size_t buf_size) = 0;
    };

  private:

    static void startTag(void *data, const char *element, const char **attribute);
    static void endTag(void *data, const char *element);
    static void characterData(void* data, const char *string, int string_len);

    std::unique_ptr<DataSource> source;

    struct State {

        unsigned lineNr, columnNr;

        std::string text;   ///< text or tag name

        NodeType type;

        std::map<std::string, std::string> attributes;

        State(NodeType type, unsigned lineNr, unsigned columnNr, const std::string& text): lineNr(lineNr), columnNr(columnNr), text(text), type(type) {}

        bool hasWhiteText() {
            for (std::size_t i = 0; i < text.size(); ++i)
                if (!isspace(text[i])) return false;
            return true;
        }
    };

    State& appendState(NodeType type, const std::string& text);

    /// parsed states (if last one is NODE_TEXT than in can be not complatly parsed)
    std::deque<State> states;

    /// xml reader, low level
    XML_Parser parser;

    /// path from root to current tag
    std::vector<std::string> path;

    /// attributes which was read in current tag
    std::unordered_set<std::string> read_attributes;

    /// true if the reader should check if there are no spurious attributes in the current element
    bool check_if_all_attributes_were_read;

    bool hasCurrent() const {
        if (states.empty()) return false;
        return states.size() > 1 || states.front().type != NODE_TEXT;
    }

    void ensureHasCurrent() const {
        if (!hasCurrent()) throw XMLException("XML reader: no current node (missing first read() call?)");
    }

    const State& getCurrent() const {
        return states.front();
    }

    inline bool strToBool(std::string& str, const std::string& name) const {
        boost::algorithm::to_lower(str);
        if (str == "yes" || str == "true" || str == "1") return true;
        else if (str == "no" || str == "false" || str == "0") return false;
        else throw XMLBadAttrException(*this, name, str);
    }

    /// @return @c true if has more data to read
    bool readSome();

    void initParser();

  public:

    /**
     * Construct XML reader to read XML from given source.
     * @param source source of XML data, will be delete by this after use by this
     */
    XMLReader(DataSource* source);
    
    /**
     * Construct XML reader to read XML from given stream.
     * @param istream stream to read, will be closed and delete by this
     */
    XMLReader(std::istream* istream);

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
    //TODO rename to next();    - like in most pull API (java)
    bool read();

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
     * @return vector of names of all opened tags, first is root, last is current tag
     */
    const std::vector<std::string>& getPath() const { return path; }

    /**
     * Get level of current node. Root has level 0, children of root have level 1, and so on.
     * @return level of current node which is equal to size of path returned by getPath()
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
     * Mark attribute with given name as read, so parser does not throw an exception if this attribute will be not read.
     * @param name name of attribute to ignore
     */
    void ignoreAttribute(const std::string& name) { getAttribute(name); }

    /**
     * Allow to have unread attributes.
     */
    void ignoreAllAttributes() { check_if_all_attributes_were_read = false; }

    /**
     * Check if current node has attribute with given @p name.
     * @param name attribute name
     * @return @c true only if current node has attribute with given @p name
     */
    bool hasAttribute(const std::string& name) const { return getAttribute(name); }

    /**
     * Returns the name of the current node.
     *
     * Throw exception if it is not defined.
     * @return name of the current node
     */
    std::string getNodeName() const { return getCurrent().text; }

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
        return boost::lexical_cast<T>(getTextContent());
    }

    /**
     * Get value of attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node.
     * @param name name of attribute
     * @param default_value default value which will be return when attribute with given @p name is not defined
     * @return attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node
     * @tparam T required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
     */
    template <typename T>
    inline T getAttribute(const std::string& name, const T& default_value) const {
        boost::optional<std::string> attr_str = getAttribute(name);
        if (attr_str) {
            return boost::lexical_cast<T>(*attr_str);
        } else
            return default_value;

        /*if (attr_str == nullptr) return default_value;
        try {
            return boost::lexical_cast<T>(attr_str);
        } catch (boost::bad_lexical_cast) {
            throw XMLBadAttrException(*this, name, attr_str);
        }*/
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
     * @tparam T required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
     */
    template <typename T>
    inline boost::optional<T> getAttribute(const std::string& name) const {
        boost::optional<std::string> attr_str = getAttribute(name);
        if (!attr_str) return boost::optional<T>();
        try {
            return boost::lexical_cast<T>(*attr_str);
        } catch (boost::bad_lexical_cast) {
            throw XMLBadAttrException(*this, name, *attr_str);
        }
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
        try {
            return boost::lexical_cast<T>(requireAttribute(name));
        } catch (boost::bad_lexical_cast) {
            throw XMLBadAttrException(*this, name, requireAttribute(name));
        }
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
     * Call requireNext() and read all text elements even if they are separated by comments. In the end require closing of a tag.
     * Throw exception no text is read or there is anything else than closing tag afterwards.
     * \return read text
     */
    std::string requireTextUntilEnd();

    /**
     * Call requireNext() and next check if current element is text. Throw exception if it's not.
     * \return read text casted (by lexical_cast) to givent type T
     */
    template <typename T>
    inline T requireText() {
        return boost::lexical_cast<T>(requireText());
    }

    /*
     * Skip XML comments.
     * @return @c true if read non-comment or @c false if XML data end
     */
    //bool skipComments();

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
     * Skip everything up to end of current tag.
     */
    void gotoEndOfCurrentTag();
};


template <>
inline bool XMLReader::getAttribute<bool>(const std::string& name, const bool& default_value) const {
    boost::optional<std::string> ostr = getAttribute(name);
    if (ostr) return strToBool(*ostr, name); else return default_value;
}

template <>
inline boost::optional<bool> XMLReader::getAttribute<bool>(const std::string& name) const {
    boost::optional<std::string> ostr = getAttribute(name);
    if (ostr) return boost::optional<bool>(strToBool(*ostr, name)); else return boost::optional<bool>();
}

template <>
inline bool XMLReader::requireAttribute<bool>(const std::string& name) const {
    boost::optional<bool> result = getAttribute<bool>(name);
    if (!result) throw XMLNoAttrException(*this, name);
    return *result;
}


}   // namespace plask


namespace std {
inline void swap(plask::XMLReader& a, plask::XMLReader& b) { a.swap(b); }
}   // namespace std

#endif // PLASK__UTILS_XML_READER_H
