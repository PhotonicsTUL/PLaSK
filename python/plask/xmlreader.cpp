#include "python_globals.h"

#include <plask/utils/xml/reader.h>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {

//     /**
//      * Get current type of node.
//      * @return current type of node
//      */
//     NodeType getNodeType() const { ensureHasCurrent(); return getCurrent().type; }
//
//     /**
//      * Get line number where current element starts.
//      * @return line number of current element start
//      */
//     unsigned getLineNr() const { ensureHasCurrent(); return getCurrent().lineNr; }
//
//     /**
//      * Get column number where current element starts.
//      * @return column number of current element start
//      */
//     unsigned getColumnNr() const { ensureHasCurrent(); return getCurrent().columnNr; }
//
//     /**
//      * Reads forward to the next xml node.
//      * @return @c false only if there is no further node.
//      */
//     bool next();
//
//     /**
//      * Get vector of names of all opened tags from root to current one.
//      * Includes tag which is just closed when the current node is NODE_ELEMENT_END.
//      * @return vector of names of all opened tags, first is root, last is current tag
//      */
//     const std::vector<std::string>& getPath() const { return path; }
//
//     /**
//      * Get level of current node. Root has level 1, children of root have level 2, and so on.
//      * @return level of current node which is equal to length of path returned by getPath()
//      */
//     std::size_t getLevel() const { return path.size(); }
//
//     /**
//      * Returns attribute count of the current XML node.
//      *
//      * This is usually non 0 if the current node is NODE_ELEMENT, and the element has attributes.
//      * @return amount of attributes of this xml node.
//      */
//     std::size_t getAttributeCount() const { ensureHasCurrent(); return getCurrent().attributes.size(); }
//
//     /**
//      * Get all attributes, empty if current node is not NODE_ELEMENT.
//      * @return all attributes, reference is valid up to read() call
//      */
//     const std::map<std::string, std::string>& getAttributes();
//
//     /**
//      * Returns the name of the current node (tag).
//      *
//      * Throw exception if it is not defined.
//      * @return name of the current node
//      */
//     std::string getNodeName() const;
//
//     /**
//      * Returns the name of the current node (tag).
//      *
//      * Throw exception if it is not defined.
//      * @return name of the current node
//      */
//     inline std::string getTagName() const { return getNodeName(); }
//
//     /**
//      * Check if current node is NODE_TEXT (throw exception if it's not) and get node data (text content).
//      * @return data of the current node
//      */
//     std::string getTextContent() const;
//
//     /**
//      * Get value of attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node.
//      * @param name name of attribute
//      * @param default_value default value which will be return when attribute with given @p name is not defined
//      * @return attribute with given @p name, or @p default_value if attribute with given @p name is not defined in current node
//      * @tparam T required type of value, boost::lexical_cast\<T> or registered parser will be used to obtain value of this type from string
//      */
//     template <typename T>
//     inline T getAttribute(const std::string& name, const T& default_value) const {
//         boost::optional<std::string> attr_str = getAttribute(name);
//         if (attr_str) {
//             return parse<T>(*attr_str, name);
//         } else
//             return default_value;
//     }
//
//     /**
//      * Get value of attribute with given @p name.
//      * @param name name of attribute to get
//      * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
//      */
//     boost::optional<std::string> getAttribute(const std::string& name) const;
//
//     /**
//      * Get value of attribute with given @p name.
//      *
//      * Throws exception if value of attribute given @p name can't be casted to required type T.
//      * @param name name of attribute to get
//      * @return boost::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
//      * @tparam T required type of value, boost::lexical_cast\<T> or registered parser will be used to obtain value of this type from string
//      */
//     template <typename T>
//     inline boost::optional<T> getAttribute(const std::string& name) const {
//         boost::optional<std::string> attr_str = getAttribute(name);
//         if (!attr_str) return boost::optional<T>();
//         return parse<T>(*attr_str, name);
//     }
//
//     /**
//      * Require the attribute with given \p name.
//      *
//      * Throws exception if there is no attribute with given \p name.
//      * \return its value
//      */
//     std::string requireAttribute(const std::string& attr_name) const;
//
//     /**
//      * Require the attribute with given \p name.
//      *
//      * Throws exception if there is no attribute with given \p name.
//      * \return its value
//      * \tparam T required type of value, boost::lexical_cast\<T> will be used to obtain value of this type from string
//      */
//     template <typename T>
//     inline T requireAttribute(const std::string& name) const {
//         return parse<T>(requireAttribute(name), name);
//     }
//
//     /**
//      * Go to next element.
//      * @throw XMLUnexpectedEndException if there is no next element
//      */
//     void requireNext();
//
//     /**
//      * Go to next element.
//      * @param required_types bit sum of NodeType-s
//      * @param new_tag_name (optional) name of required tag (ingored if NODE_ELEMENT is not included in required_types)
//      * @return type of new current node
//      * @throw XMLUnexpectedElementException if node type is not included in @p required_types or
//      * (only when new_tag_name is given) node type is @c NODE_ELEMENT and name is not equal to @p new_tag_name.
//      * @throw XMLUnexpectedEndException if there is no next element
//      */
//     NodeType requireNext(int required_types, const char *new_tag_name = nullptr);
//
//     /**
//      * Call requireNext() and next check if current element is tag opening. Throw exception if it's not.
//      */
//     void requireTag();
//
//     /**
//      * Call requireNext() and next check if current element is tag opening.
//      * Throw exception if it's not or if it name is not \p name.
//      */
//     void requireTag(const std::string& name);
//
//     /**
//      * Call requireNext() and next check if current element is tag opening or closing of tag.
//      * Throw exception if it's not.
//      * \return true if the next tag was opened
//      */
//     bool requireTagOrEnd();
//
//     /**
//      * Call requireNext() and next check if current element is tag opening (in such case it also check if it has name equal to given @p name) or closing of tag.
//      * Throw exception if it's not.
//      * @param name required name of opening tag
//      * @return true if the next tag was opened
//      */
//     bool requireTagOrEnd(const std::string &name);
//
//     /**
//      * Call requireNext() and next check if current element is tag closing. Throw exception if it's not.
//      */
//     void requireTagEnd();
//
//     /**
//      * Call requireNext() and next check if current element is text. Throw exception if it's not.
//      * \return read text
//      */
//     std::string requireText();
//
//     /**
//      * Read text inside current tag. Move parser to end of current tag.
//      * \return read text
//      */
//     std::string requireTextInCurrentTag();
//
//     /**
//      * Call requireNext() and next check if current element is text. Throw exception if it's not. Next require end of tag.
//      * \return read text casted (by lexical_cast) to givent type T
//      */
//     template <typename T>
//     inline T requireTextInCurrentTag() {
//         return parse<T>(requireTextInCurrentTag());
//     }
//
//     /**
//      * Skip everything up to element with required type on required level.
//      * @param required_level level on which required element should be
//      * @param required_type type of required element
//      * @return @c true if reader is on required element or @c false if XML data end
//      */
//     bool gotoNextOnLevel(std::size_t required_level, NodeType required_type = NODE_ELEMENT);
//
//     /**
//      * Skip everything up to next tag element on current level.
//      * @return @c true if reader is on required element or @c false if XML data end
//      */
//     bool gotoNextTagOnCurrentLevel();
//
//     /**
//      * Skip everything up to end of the current tag.
//      */
//     void gotoEndOfCurrentTag();

/*
 * for tag in xmlreader:
 *     if tag.name == "something":
 *         for sub in tag:
 *             if sub.name == "withtext":
 *                 text = sub.text
 *             else:
 *                 a = tag[a]
 *                 b = tag.get(b)
 *     else:
 *         self.read_xml_tag(tag, manager)
 *
*/

namespace detail {

    class XMLIterator {

        XMLReader* reader;
        size_t level;

        inline size_t current_level() {
            return reader->getLevel() - size_t(reader->getNodeType() == XMLReader::NODE_ELEMENT_END);
        }

      public:

        XMLIterator(XMLReader* reader): reader(reader), level(reader->getLevel()) {}

        XMLReader* next() {
            for (size_t i = current_level(); i > level; --i) { reader->requireTagEnd(); }
            if (!reader->requireTagOrEnd()) {
                PyErr_SetString(PyExc_StopIteration, "");
                py::throw_error_already_set();
            }
            return reader;
        }
    };

    static XMLIterator XMLReader__iter__(XMLReader* reader) {
        return XMLIterator(reader);
    }

    static py::object XMLReader__getitem__(XMLReader* reader, const std::string& key) {
        auto value = reader->requireAttribute(key);
        py::str obj(value);
        try {
            return py::eval(obj);
        } catch (py::error_already_set) {
            PyErr_Clear();
            return obj;
        }
    }

    static py::object XMLReader_get(XMLReader* reader, const std::string& key, const py::object& deflt) {
        auto value = reader->getAttribute(key);
        if (value) {
            py::str obj(*value);
            try {
                return py::eval(obj);
            } catch (py::error_already_set) {
                PyErr_Clear();
                return obj;
            }
        }
        else return deflt;
    }

}

void register_xml_reader() {

    py::class_<XMLReader, XMLReader*, boost::noncopyable> xml("XmlReader", py::no_init); xml
        .def("__iter__", &detail::XMLReader__iter__)
        .add_property("name", &XMLReader::getNodeName, "Current tag name.")
        .add_property("text", (std::string(XMLReader::*)())&XMLReader::requireTextInCurrentTag, "Text in the current tag.")
        .def("__getitem__", &detail::XMLReader__getitem__)
        .def("get", detail::XMLReader_get, "Return tag attribute value or default if the attribute does not exist.",
             (py::arg("key"), py::arg("default")=py::object()))
    ;

    py::scope scope(xml);

    py::class_<detail::XMLIterator>("_Iterator", py::no_init)
        .def(NEXT, &detail::XMLIterator::next, py::return_value_policy<py::reference_existing_object>())
    ;
}


}} // namespace plask::python
