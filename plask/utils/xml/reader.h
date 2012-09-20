#ifndef PLASK__UTILS_XML_READER_H
#define PLASK__UTILS_XML_READER_H

#include <irrxml/irrXML.h>

#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <unordered_set>

#include "exceptions.h"

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
            NODE_NONE = irr::io::EXN_NONE,                  ///< No xml node. This is usually the node if you did not read anything yet.
            NODE_ELEMENT = irr::io::EXN_ELEMENT,            ///< A xml element, like \<foo>
            NODE_ELEMENT_END = irr::io::EXN_ELEMENT_END,    ///< End of an xml element, like \</foo>
            NODE_TEXT = irr::io::EXN_TEXT,                  ///< Text within a xml element: \<foo> this is the text. \</foo>
            NODE_COMMENT =  irr::io::EXN_COMMENT,           ///< An xml comment like &lt;!-- I am a comment --&gt; or a DTD definition.
            NODE_CDATA =  irr::io::EXN_CDATA,               ///< An xml cdata section like &lt;![CDATA[ this is some CDATA ]]&gt;
            NODE_UNKNOWN = irr::io::EXN_UNKNOWN             ///< Unknown element
    };

  private:
    /// xml reader, low level
    irr::io::IrrXMLReader* irrReader;

    /// current type of node
    NodeType currentNodeType;

    /// path from root to current tag
    std::vector<std::string> path;

    /// attributes which was read
    std::unordered_set<std::string> read_attributes;

  public:

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

    /**
     * Construct XML reader to read XML from given @p input stream.
     * @param input input stream
     */
    XMLReader(std::istream& input);

#if (__cplusplus >= 201103L) || defined(__GXX_EXPERIMENTAL_CXX0X__)
    /**
     * Move constructor.
     * @param to_move object to move from, should not be used but only delete after move
     */
    XMLReader(XMLReader&& to_move);

    /**
     * Move assigment operator.
     * @param to_move object to move from, should not be used but only delete after move
     * @return *this
     */
    XMLReader& operator=(XMLReader&& to_move);

    /// Disallow copy of reader.
    XMLReader(const XMLReader& to_copy) = delete;
    XMLReader& operator=(const XMLReader& to_copy) = delete;
#else
  private:
    XMLReader(const XMLReader&) {}
    XMLReader& operator=(const XMLReader&) { return *this; }
  public:
#endif

    ~XMLReader() { delete irrReader; }  //if irrReader is not nullptr throw exception if path is not empty

    /**
     * Swap states of @c this and @p to_swap.
     * @param to_swap object to swap with this
     */
    void swap(XMLReader& to_swap);

    /**
     * Get current type of node.
     * @return current type of node
     */
    NodeType getNodeType() const { return currentNodeType; }

    /**
     * Reads forward to the next xml node.
     * @param check_if_all_attributes_was_read if @c true (default) and current tag is NODE_ELEMENT, parser will check if all attributes was read from it and throw excpetion if it was not
     * @return @c false only if there is no further node.
    */
    bool read(bool check_if_all_attributes_was_read = true);

    /**
     * Check if node is empty, like \<foo /\>.
     *
     * Note that empty nodes are comunicate by parser two times: as NODE_ELEMENT and next as NODE_ELEMENT_END.
     * So for \<foo /\> parser work just like for \<foo>\</foo> and only this method allow to check which notation was used.
     * @return if an element is an empty element, like \<foo /\>
     */
    bool isEmptyElement() const { return irrReader->isEmptyElement(); }

    /**
     * Get vector of names of all opened tags from root to current one.
     * @return vector of names of all opened tags, first is root, last is current tag
     */
    const std::vector<std::string> getPath() const { return path; }
    
    /**
     * Get level of current node. Root has level 0, children of root have level 1, and so on.
     * @return level of current node which is equal to size of path returned by getPath()
     */
    std::size_t getLevel() const { return path.size(); }

    /**
     * Returns attribute count of the current XML node.

     * This is usually non 0 if the current node is NODE_ELEMENT, and the element has attributes.
     * @return amount of attributes of this xml node.
     */
    int getAttributeCount() const { return irrReader->getAttributeCount(); }

    /* Returns name of an attribute.
    @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
    @return Name of the attribute, 0 if an attribute with this index does not exist. */
    //const char* getAttributeNameC(int idx) const { return irrReader->getAttributeName(idx); }

    /* Returns the value of an attribute.
     @param idx: Zero based index, should be something between 0 and getAttributeCount()-1.
     @return Value of the attribute, 0 if an attribute with this index does not exist. */
    //const char* getAttributeValueC(int idx) const { return irrReader->getAttributeValue(idx); }

    /**
     * Returns the value of an attribute.
     * @param name: Name of the attribute.
     * @return Value of the attribute, 0 if an attribute with this name does not exist.
     */
    const char* getAttributeValueC(const std::string& name) const;

    /**
     * Mark argument with given name as read, so parser not throw an exeption if this attribute will be not read.
     * @param name name of attribute to ignore
     */
    void ignoreAttribute(const std::string& name) { getAttributeValueC(name); }

    /* Returns the value of an attribute in a safe way.

    Like getAttributeValue(), but does not
    return 0 if the attribute does not exist. An empty string ("") is returned then.
    @param name: Name of the attribute.
    @return Value of the attribute, and "" if an attribute with this name does not exist */
    //const char* getAttributeValueOrEmptyC(const char* name) const { return irrReader->getAttributeValueSafe(name); }

    //bool hasAttribute(const char* name) const { return getAttributeValueC(name) != 0; }

    /**
     * Check if current node has attribute with given @p name.
     * @param name attribute name
     * @return @c true only if current node has attribute with given @p name
     */
    bool hasAttribute(const std::string& name) const { return getAttributeValueC(name) != 0; }

    /**
     * Returns the name of the current node.
     *
     * Only non null, if the node type is NODE_ELEMENT.
     * @return name of the current node or 0 if the node has no name.
     */
    const char* getNodeNameC() const { return irrReader->getNodeName(); }

    /**
     * Returns the name of the current node.
     *
     * Bechaviour is undefined if name is not defined.
     * @return name of the current node
     */
    std::string getNodeName() const { return irrReader->getNodeName(); }

    /** Returns data of the current node.
     *
     * Only non null if the node has some data and it is of type NODE_TEXT or NODE_UNKNOWN.
     * @return data of the current node
     */
    const char* getNodeDataC() const { return irrReader->getNodeData(); }

    /**
     * Check if current node is NODE_TEXT (throw excpetion if it's not) and get node data (text content).
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
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return default_value;
        return boost::lexical_cast<T>(attr_str);
    }

    /**
     * Get value of attribute with given @p name.
     * @param name name of attribute to get
     * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribiute with given @p name
     */
    boost::optional<std::string> getAttribute(const std::string& name) const;

    /**
     * Get value of attribute with given @p name.
     *
     * Throws exception if value of attribute given @p name can't be casted to required type T.
     * @param name name of attribute to get
     * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribiute with given @p name
     * @tparam T required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
     */
    template <typename T>
    inline boost::optional<T> getAttribute(const std::string& name) const {
        const char* attr_str = getAttributeValueC(name);
        if (attr_str == nullptr) return boost::optional<T>();
        return boost::lexical_cast<T>(attr_str);
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
        return boost::lexical_cast<T>(requireAttribute(name));
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
     * Call requireNext() and next check if current element is text. Throw exception if it's not.
     * \return read text casted (by lexical_cast) to givent type T
     */
    template <typename T>
    inline T requireText() {
        return boost::lexical_cast<T>(requireText());
    }

    /**
     * Skip XML comments.
     * @return @c true if read non-comment or @c false if XML data end
     */
    bool skipComments();
    
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
    const char* cstr = getAttributeValueC(name);
    if (cstr != nullptr) {
        std::string str(cstr);
        boost::algorithm::to_lower(str);
        if (str == "yes" || str == "true" || str == "1") return true;
        else if (str == "no" || str == "false" || str == "0") return false;
        else throw XMLBadAttrException(*this, name, str);
    }
    return default_value;
}

template <>
inline boost::optional<bool> XMLReader::getAttribute<bool>(const std::string& name) const {
    const char* cstr = getAttributeValueC(name);
    if (cstr != nullptr) {
        std::string str(cstr);
        boost::algorithm::to_lower(str);
        if (str == "yes" || str == "true" || str == "1") return true;
        else if (str == "no" || str == "false" || str == "0") return false;
        else throw XMLBadAttrException(*this, name, str);
    }
    return boost::optional<bool>();
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
