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
#include "exceptions.hpp"
#include "reader.hpp"

namespace plask {

XMLException::XMLException(const XMLReader& reader, const std::string& msg):
    std::runtime_error("XML line " + boost::lexical_cast<std::string>(reader.getLineNr()) +
                       ((reader.getNodeType() == XMLReader::NODE_ELEMENT)? " in <" + reader.getNodeName() + ">" :
                        (reader.getNodeType() == XMLReader::NODE_ELEMENT_END)? " in </" + reader.getNodeName() + ">" :
                        "") +
                       ": " + msg), line(reader.getLineNr()) {}

XMLException::XMLException(const std::string& where, const std::string& msg, int line):
    std::runtime_error(where + ": " + msg), line(line) {}

XMLException::XMLException(const std::string& msg, int line): std::runtime_error(msg), line(line) {}

XMLUnexpectedElementException::XMLUnexpectedElementException(const XMLReader& reader, const std::string& what_is_expected):
    XMLException(reader, "expected " + what_is_expected + ", got " + (
                    reader.getNodeType() == XMLReader::NODE_ELEMENT ?     ("<"+reader.getNodeName()+">") :
                    reader.getNodeType() == XMLReader::NODE_ELEMENT_END ? ("</"+reader.getNodeName()+">") :
                    "text"
                 ) + " instead") {}


} // namespace plask
