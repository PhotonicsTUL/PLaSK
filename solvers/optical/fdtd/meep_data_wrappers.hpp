#ifndef MEEP_DATA_WRAPPERS_H
#define MEEP_DATA_WRAPPERS_H

#include <meep.hpp>
#include <plask/plask.hpp>
#include <plask/python.hpp>
#include "comp_map.cpp"

namespace plask { namespace solvers { namespace optical_fdtd {


namespace dft {

template <typename T> class MeepHasNewDftFlux {
    template <typename C> static std::false_type test(const decltype(&C::Nfreq)*);
    template <typename C> static std::false_type test(const decltype(&C::Nomega)*);
    template <typename C> static std::true_type test(const decltype(&C::freq)*);
    template <typename C> static std::true_type test(const decltype(&C::omega)*);

  public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};


template <typename T>
inline static typename std::enable_if<MeepHasNewDftFlux<T>::value, std::vector<double>>::type wavelengths(const T& data) {
    std::vector<double> wavelengths;
    wavelengths.reserve(data.freq.size());
    for (double freq : data.freq) wavelengths.push_back(1e3 / freq);
    return wavelengths;
}


template <typename T>
inline static typename std::enable_if<!MeepHasNewDftFlux<T>::value, std::vector<double>>::type wavelengths(const T& data) {
    std::vector<double> wavelengths;
    auto nfreq = data.Nfreq;
    auto dfreq = data.dfreq;
    auto min_freq = data.freq_min;
    wavelengths.reserve(nfreq);
    for (size_t i = 0; i < nfreq; ++i) wavelengths.push_back(1e3 / (min_freq + i * dfreq));
    return wavelengths;
}


template <typename T> inline static typename std::enable_if<MeepHasNewDftFlux<T>::value, size_t>::type freq_size(const T& data) {
    return data.freq.size();
}

template <typename T> inline static typename std::enable_if<!MeepHasNewDftFlux<T>::value, size_t>::type freq_size(const T& data) {
    return data.Nfreq;
}

template <typename T> inline static typename std::enable_if<MeepHasNewDftFlux<T>::value, double>::type freq0(const T& data) {
    return data.freq[0];
}

template <typename T> inline static typename std::enable_if<!MeepHasNewDftFlux<T>::value, double>::type freq0(const T& data) {
    return data.freq_min;
}


template <typename T> inline static typename std::enable_if<MeepHasNewDftFlux<T>::value, size_t>::type omega_size(const T* chunk) {
    return chunk->omega.size();
}

template <typename T> inline static typename std::enable_if<!MeepHasNewDftFlux<T>::value, size_t>::type omega_size(const T* chunk) {
    return chunk->Nomega;
}

}

struct SourceMEEP {
    std::string source_type;
    double ax, ay, bx, by, lam, start_time, slowness, amplitude, width;
    std::string component, end_time;
    SourceMEEP(std::string sType,
               double ax,
               double ay,
               double bx,
               double by,
               double lam,
               double start_time,
               std::string end_time,
               double amplitude,
               double width,
               double slowness,
               std::string comp);
};


struct HarminvRawData {
    shared_ptr<std::vector<dcomplex>> amp;
    shared_ptr<std::vector<double>> freq_re, freq_im, freq_err;
    HarminvRawData() = default;
    HarminvRawData(shared_ptr<std::vector<dcomplex>> a,
                   shared_ptr<std::vector<double>> fr,
                   shared_ptr<std::vector<double>> fim,
                   shared_ptr<std::vector<double>> ferr)
        : amp(a), freq_re(fr), freq_im(fim), freq_err(ferr) {}
};

struct HarminvValue {
    int k;
    shared_ptr<HarminvRawData> data;
    HarminvValue() = default;
    HarminvValue(int k, const shared_ptr<HarminvRawData> data) : k(k), data(data){};

    dcomplex omega() { return dcomplex((*(data->freq_re))[k], (*(data->freq_im))[k]); }
    double omega_err() { return (*(data->freq_err))[k]; }
    dcomplex amp() { return (*(data->amp))[k]; }
    double amp_mag() { return abs(amp()); }
    double q_factor() { return ((*(data->freq_re))[k]) / (-2 * ((*(data->freq_im))[k])); }
    double wavelength() { return 1 / (1e-3 * ((*(data->freq_re))[k])); }
};

struct HarminvResults {
    shared_ptr<HarminvRawData> data;
    int n_res;

    HarminvResults() = default;
    HarminvResults(shared_ptr<HarminvRawData> data, int n) : data(data), n_res(n) {}

    int len() const { return n_res; }

    HarminvValue getItem(int k) {
        int n = len();
        if (k < 0) k = n - k;
        if (k < 0 || k >= n) throw plask::python::IndexError("Value outside of the possible index.");
        return HarminvValue(k, data);
    }
};

struct FieldsDFT {
    shared_ptr<meep::dft_fields> data;
    shared_ptr<meep::fields> fields;

    FieldsDFT() = default;
    FieldsDFT(shared_ptr<meep::dft_fields> data, shared_ptr<meep::fields> fields) : data(data), fields(fields) {}

    double freq() {
        return dft::freq0(*data);
    }

    std::vector<dcomplex> field(const std::string& comp) {
        int rank = 2;
        size_t dims[] = {1, 1, 0};
        meep::component component = ComponentMap.at(comp);
        auto first_elem = fields->get_dft_array(*data, component, 1, &rank, dims);
        return std::vector<dcomplex>{dcomplex{1., 10.}};
    }
};

struct FluxDFT {  // Let it be a temporary name for now
    shared_ptr<meep::dft_flux> data;

    FluxDFT() = default;
    FluxDFT(shared_ptr<meep::dft_flux> data) : data(data){};

    int n_wave() const {
        return dft::freq_size(*data);
    }

    std::vector<double> get_flux() {
        auto first = data->flux();
        std::vector<double> flux_vector;
        for (int i = 0, n = n_wave(); i < n; ++i) flux_vector.push_back(*first++);
        return flux_vector;
    }

    std::vector<double> get_waves() {
        return dft::wavelengths(*data);
    }

    void load_minus_flux_data(const FluxDFT& loaded_flux) {
        // Load the transformed field data into a vector
        std::vector<dcomplex> e_loaded_field, h_loaded_field;
        int i = 0;
        for (meep::dft_chunk* current_chunk = loaded_flux.data->E; current_chunk; current_chunk = current_chunk->next_in_dft) {
            size_t steps = dft::omega_size(current_chunk) * (current_chunk->N);
            for (size_t i = 0; i < steps; ++i) {
                e_loaded_field.push_back(current_chunk->dft[i]);
            }
        }

        for (meep::dft_chunk* current_chunk = loaded_flux.data->H; current_chunk; current_chunk = current_chunk->next_in_dft) {
            size_t steps = dft::omega_size(current_chunk) * (current_chunk->N);
            for (size_t i = 0; i < steps; ++i) {
                h_loaded_field.push_back(current_chunk->dft[i]);
            }
        }

        // Copy the values into the flux
        size_t start_pos = 0;
        for (meep::dft_chunk* current_chunk = data->E; current_chunk; current_chunk = current_chunk->next_in_dft) {
            size_t steps = dft::omega_size(current_chunk) * (current_chunk->N);
            for (size_t i = 0; i < steps; ++i) {
                data->E->dft[i] = e_loaded_field[i + start_pos];
            }
            start_pos += steps;
        }

        start_pos = 0;
        for (meep::dft_chunk* current_chunk = data->H; current_chunk; current_chunk = current_chunk->next_in_dft) {
            size_t steps = dft::omega_size(current_chunk) * (current_chunk->N);
            for (size_t i = 0; i < steps; ++i) {
                data->H->dft[i] = h_loaded_field[i + start_pos];
            }
            start_pos += steps;
        }

        // Negate the transformed fields
        data->scale_dfts(-1);
    }
};

}}}  // namespace plask::solvers::optical_fdtd

#endif
