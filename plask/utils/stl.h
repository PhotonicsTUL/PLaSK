#ifndef PLASK__STL_H
#define PLASK__STL_H

namespace plask {

template <typename map_t>
inline typename map_t::mapped_type map_find(map_t& map, const typename map_t::key_type& to_find, typename map_t::mapped_type if_not_found = nullptr) {
	typename map_t::iterator f = map.find(to_find);
	return f == map.end() ? if_not_found : f->second;
}

template <typename map_t>
inline const typename map_t::mapped_type map_find(const map_t& map, const typename map_t::key_type& to_find, const typename map_t::mapped_type if_not_found = nullptr) {
	typename map_t::const_iterator f = map.find(to_find);
	return f == map.end() ? if_not_found : f->second;
}

}	// namespace plask

#endif // PLASK__STL_H
