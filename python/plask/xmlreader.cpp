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
#include <limits>
#include <boost/algorithm/string.hpp>

#include "plask/utils/xml/reader.hpp"

#include "python_globals.hpp"

namespace plask { namespace python {

extern PLASK_PYTHON_API std::string xplFilename;


void removeIndent(std::string& text, unsigned xmlline, const char* tag) {
    auto line =  boost::make_split_iterator(text, boost::token_finder(boost::is_any_of("\n"), boost::token_compress_off));
    const boost::algorithm::split_iterator<std::string::iterator> endline;
    auto firstline = line;
    size_t strip;
    std::string::iterator beg;
    bool cont;
    do { // Search for the first non-empty line to get initial indentation
        strip = 0;
        for (beg = line->begin(); beg != line->end() && (*beg == ' ' || *beg == '\t'); ++beg) {
            if (*beg == ' ') ++strip;
            else { strip += 8; strip -= strip % 8; } // add to closest full tab-stop
        }
        cont = beg == line->end();
        line++;
    } while (cont && line != endline);
    if (beg == line->begin() || line == endline) {
        boost::trim_left(text);
        return;
    }
    std::string result;
    line = firstline;
    for (size_t lineno = 1; line != endline; ++line, ++lineno) { // indent all lines
        size_t pos = 0;
        for (beg = line->begin(); beg != line->end() && (pos < strip); ++beg) {
            if (*beg == ' ') ++pos;
            else if (*beg == '\t') { pos += 8; pos -= pos % 8; } // add to closest full tab-stop
            else if (*beg == '#') { break; } // allow unidentation for comments
            else {
                ptrdiff_t d = std::distance(line->begin(), beg);
                throw XMLException(format(u8"XML line {0}{5}: Python line indentation ({1} space{2}) is less than the indentation of the first line ({3} space{4})",
                                          xmlline+lineno, d, (d==1)?"":"s", strip, (strip==1)?"":"s",
                                          tag? format(" in <{}>", tag) : ""));
            }
        }
        result += std::string(beg, line->end());
        result += "\n";
    }
    text = std::move(result);
}


PyCodeObject* compilePythonFromXml(XMLReader& reader, Manager& manager, bool exec) {
    size_t lineno  = reader.getLineNr();
    const std::string tag = reader.getNodeName();
    const std::string name = xplFilename.empty()? format("<{}>", tag) : format("{} in <{}>, XML", xplFilename, tag);

    std::string text = reader.requireTextInCurrentTag();
    boost::trim_right_if(text, boost::is_any_of(" \n\r\t"));

    PyObject* result;
    {
        const char* s = text.c_str();
        size_t i = 0;
        while (std::isspace(*s)) {
            if (*s == '\n') lineno++;
            ++i; ++s;
        }
        result = Py_CompileString((std::string(lineno, '\n') + text.substr(i)).c_str(), name.c_str(), Py_eval_input);
    }
    if (result == nullptr && exec) {
        PyErr_Clear();
        size_t start;
        if (text.find('\n') == std::string::npos) {
            boost::trim_left(text);
        } else {
            for (start = 0; text[start] != '\n' && start < text.length(); ++start) {
                if (!std::isspace(text[start]))
                    throw XMLException(format("XML line {}", lineno),
                        format("Python code must be either a single line or must begin from new line after <{}>", tag), lineno);
            }
            if (start != text.length()) text = text.substr(start+1);
            removeIndent(text, lineno, tag.c_str());
        }
        result = Py_CompileString((std::string(lineno-1, '\n') + text).c_str(), name.c_str(), Py_file_input);
    }
    if (result == nullptr) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        Py_XDECREF(ptraceback);
        std::string type, value;
        size_t errline = reader.getLineNr();
        if (ptype) { type = py::extract<std::string>(py::object(py::handle<>(ptype)).attr("__name__")); }
        if (pvalue && PyTuple_Check(pvalue) && PyTuple_Size(pvalue) > 1) {
            py::extract<std::string> evalue(PyTuple_GetItem(pvalue, 0));
            if (evalue.check()) value = ": " + evalue();
            PyObject* pdetails = PyTuple_GetItem(pvalue, 1);
            if (pdetails && PyTuple_Check(pdetails) && PyTuple_Size(pdetails) > 1) {
                py::extract<int> line(PyTuple_GetItem(pdetails, 1));
                if (line.check()) errline = line();
            }
        }
        Py_XDECREF(pvalue);
        PyErr_Clear();
        manager.throwErrorIfNotDraft(
            XMLException(format("XML line {} in <{}>", errline, tag), format("{}{}", type, value), errline));
        return nullptr;
    }
    return reinterpret_cast<PyCodeObject*>(result);
}

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
//         plask::optional<std::string> attr_str = getAttribute(name);
//         if (attr_str) {
//             return parse<T>(*attr_str, name);
//         } else
//             return default_value;
//     }
//
//     /**
//      * Get value of attribute with given @p name.
//      * @param name name of attribute to get
//      * @return plask::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
//      */
//     plask::optional<std::string> getAttribute(const std::string& name) const;
//
//     /**
//      * Get value of attribute with given @p name.
//      *
//      * Throws exception if value of attribute given @p name can't be casted to required type T.
//      * @param name name of attribute to get
//      * @return plask::optional which represent value of attribute with given @p name or has no value if there is no attribute with given @p name
//      * @tparam T required type of value, boost::lexical_cast\<T> or registered parser will be used to obtain value of this type from string
//      */
//     template <typename T>
//     inline plask::optional<T> getAttribute(const std::string& name) const {
//         plask::optional<std::string> attr_str = getAttribute(name);
//         if (!attr_str) return plask::optional<T>();
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
        return eval_common_type(reader->requireAttribute(key));
    }

    static py::object XMLReader_get(XMLReader* reader, const std::string& key, const py::object& deflt) {
        auto value = reader->getAttribute(key);
        if (value) return eval_common_type(*value);
        else return deflt;
    }

    static py::object XMLReader_getitem(XMLReader* reader, const py::object& dict, const std::string& key) {
        return dict[eval_common_type(reader->requireAttribute(key))];
    }

    static py::object XMLReader_attribs(XMLReader* reader) {
        py::dict result;
        for (auto attr: reader->getAttributes())
            result[attr.first] = eval_common_type(attr.second);
        return result;
    }

    static bool XMLReader__eq__(XMLReader* reader, const py::object& other) {
        py::extract<std::string> name(other);
        if (name.check()) return reader->getNodeName() == name();
        py::extract<XMLReader*> other_reader(other);
        if (other_reader.check()) return reader == other_reader();
        return false;
    }

    static std::string XMLReader__str__(const XMLReader& reader) {
     return "XML line " + boost::lexical_cast<std::string>(reader.getLineNr()) +
            ((reader.getNodeType() == XMLReader::NODE_ELEMENT)? " in <" + reader.getNodeName() + ">" :
            (reader.getNodeType() == XMLReader::NODE_ELEMENT_END)? " in </" + reader.getNodeName() + ">" :
            "");
    }

    static std::string XMLReader__repr__(const XMLReader* self) {
        std::stringstream out;
        out << "<plask.XplReader object at (" << self << ")>";
        return out.str();
    }
}

void register_xml_reader() {

    py::class_<XMLReader, XMLReader*, boost::noncopyable> xml("XplReader", py::no_init); xml
        .def("__iter__", &detail::XMLReader__iter__)
        .def("__eq__", &detail::XMLReader__eq__)
        .add_property("name", &XMLReader::getNodeName, u8"Current tag name.")
        .add_property("text", (std::string(XMLReader::*)())&XMLReader::requireTextInCurrentTag, u8"Text in the current tag.")
        .def("__getitem__", &detail::XMLReader__getitem__)
        .def("get", detail::XMLReader_get, u8"Return tag attribute value or default if the attribute does not exist.",
             (py::arg("key"), py::arg("default")=py::object()))
        .def("getitem", detail::XMLReader_getitem, u8"Return tag attribute value as raw string or default if the attribute does not exist.",
             (py::arg("key"), py::arg("default")=""))
        .add_property("attrs", &detail::XMLReader_attribs, u8"List of all the tag attributes.")
        .def("__contains__", &XMLReader::hasAttribute)
        .def("__str__", detail::XMLReader__str__)
        .def("__repr__", detail::XMLReader__repr__)
    ;

    py::scope scope(xml);
    (void) scope;   // don't warn about unused variable scope

    py::class_<detail::XMLIterator>("_Iterator", py::no_init)
        .def("__next__", &detail::XMLIterator::next, py::return_value_policy<py::reference_existing_object>())
    ;
}


}} // namespace plask::python
