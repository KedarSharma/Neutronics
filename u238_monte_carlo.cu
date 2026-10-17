#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr float kMinimumEnergyEv = 1.0e-5f;
constexpr float kMaximumEnergyEv = 1.0e8f;
constexpr float kFissionRatePerMev = 0.776f;
constexpr int kChannelCount = 4;
constexpr int kCountersPerSpecies = 7;

enum Channel : int {
    Elastic = 0,
    Inelastic = 1,
    Capture = 2,
    Fission = 3,
};

enum FissionRegion : int {
    ThermalRegion = 0,
    ResonanceRegion = 1,
    FastRegion = 2,
};

#if defined(D2O_URANIUM)
constexpr int kSpeciesCount = 4;
constexpr int kCounterCount = kSpeciesCount * kCountersPerSpecies;
#elif defined(H2O_URANIUM)
constexpr int kSpeciesCount = 4;
constexpr int kCounterCount = kSpeciesCount * kCountersPerSpecies;
#elif defined(NATURAL_URANIUM)
constexpr int kSpeciesCount = 2;
constexpr int kCounterCount = kSpeciesCount * kCountersPerSpecies;
#else
constexpr int kSpeciesCount = 1;
constexpr int kCounterCount = kCountersPerSpecies;
#endif

__host__ __device__ constexpr int speciesMassNumber(int species) {
#if defined(D2O_URANIUM)
    constexpr int mass_numbers[kSpeciesCount] = {2, 16, 235, 238};
    return mass_numbers[species];
#elif defined(H2O_URANIUM)
    constexpr int mass_numbers[kSpeciesCount] = {1, 16, 235, 238};
    return mass_numbers[species];
#elif defined(NATURAL_URANIUM)
    return species == 0 ? 235 : 238;
#else
    return 238;
#endif
}

const char* speciesName(int species) {
#if defined(D2O_URANIUM)
    constexpr const char* names[kSpeciesCount] = {"H-2", "O-16", "U-235", "U-238"};
    return names[species];
#elif defined(H2O_URANIUM)
    constexpr const char* names[kSpeciesCount] = {"H-1", "O-16", "U-235", "U-238"};
    return names[species];
#elif defined(NATURAL_URANIUM)
    return species == 0 ? "U-235" : "U-238";
#else
    return "U-238";
#endif
}

struct CrossSectionPoint {
    double energy_ev;
    double cross_section_b;
};

struct DeviceCrossSectionTable {
    const CrossSectionPoint* points;
    std::uint32_t count;
};

struct Track {
    float energy_ev;
    std::uint32_t id;
    std::uint8_t primary;
};

struct HistoryPoint {
    std::uint32_t track_id;
    std::uint32_t collision_count;
    float energy_ev;
};

struct FinalNeutron {
    std::uint32_t track_id;
    std::uint32_t collision_count;
    float energy_ev;
    std::uint8_t terminal_channel;
};

struct GenerationNeutron {
    std::uint32_t track_id;
    float energy_ev;
};

static_assert(sizeof(HistoryPoint) == 12);
static_assert(sizeof(CrossSectionPoint) == 16);

__device__ float clampEnergy(float energy_ev) {
    return fminf(kMaximumEnergyEv, fmaxf(kMinimumEnergyEv, energy_ev));
}

__device__ float sampleBirthEnergyEv(curandStatePhilox4_32_10_t& state) {
    const float x = curand_normal(&state);
    const float y = curand_normal(&state);
    const float z = curand_normal(&state);
    const float energy_mev = (x * x + y * y + z * z) / (2.0f * kFissionRatePerMev);
    return clampEnergy(energy_mev * 1.0e6f);
}

__device__ float interpolateCrossSection(
    float energy_ev,
    const DeviceCrossSectionTable& table) {
    if (table.count == 0 ||
        energy_ev < static_cast<float>(table.points[0].energy_ev)) {
        return 0.0f;
    }

    std::uint32_t low = 0;
    std::uint32_t high = table.count;
    while (low < high) {
        const std::uint32_t middle = low + (high - low) / 2;
        if (table.points[middle].energy_ev < energy_ev) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    if (low == table.count) {
        const CrossSectionPoint& last = table.points[table.count - 1];
        if (table.count == 1) {
            return last.cross_section_b;
        }
        std::uint32_t previous_index = table.count - 2;
        while (previous_index > 0 &&
               table.points[previous_index].energy_ev == last.energy_ev) {
            --previous_index;
        }
        const CrossSectionPoint& previous = table.points[previous_index];
        const float slope = logf(last.cross_section_b / previous.cross_section_b) /
                            logf(last.energy_ev / previous.energy_ev);
        return static_cast<float>(
            last.cross_section_b *
            pow(static_cast<double>(energy_ev) / last.energy_ev, slope));
    }

    const CrossSectionPoint& upper = table.points[low];
    if (upper.energy_ev == energy_ev || low == 0) {
        return upper.cross_section_b;
    }

    const CrossSectionPoint& lower = table.points[low - 1];
    const double fraction =
        log(static_cast<double>(energy_ev) / lower.energy_ev) /
        log(upper.energy_ev / lower.energy_ev);
    return static_cast<float>(
        exp(log(lower.cross_section_b) +
            fraction * log(upper.cross_section_b / lower.cross_section_b)));
}

__device__ bool appendHistory(
    HistoryPoint* history,
    std::uint64_t history_capacity,
    unsigned long long* history_count,
    unsigned int* history_overflow,
    std::uint32_t track_id,
    std::uint32_t collision_count,
    float energy_ev) {
    const unsigned long long slot = atomicAdd(history_count, 1ULL);
    if (slot >= history_capacity) {
        atomicAdd(history_overflow, 1U);
        return false;
    }
    history[slot] = {track_id, collision_count, energy_ev};
    return true;
}

__device__ std::uint32_t fissionMultiplicity(
    curandStatePhilox4_32_10_t& state,
    int mass_number) {
    const double uniform =
        static_cast<double>(curand(&state)) * (1.0 / 4294967296.0);
    if (mass_number == 235) {
        return static_cast<std::uint32_t>(2.5 + uniform);
    }
    return static_cast<std::uint32_t>(2.43 + uniform);
}

__global__ void initializePopulation(
    Track* queue,
    curandStatePhilox4_32_10_t* random_states,
    std::uint32_t initial_count,
    unsigned long long seed) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= initial_count) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, index, 0, &state);
    const float energy = sampleBirthEnergyEv(state);
    queue[index] = {energy, index, 1};
    random_states[index] = state;
}

__global__ void simulatePopulation(
    const Track* current_queue,
    std::uint32_t current_count,
    Track* next_queue,
    std::uint64_t track_capacity,
    curandStatePhilox4_32_10_t* random_states,
    unsigned long long seed,
    std::uint32_t max_collisions_per_track,
    const DeviceCrossSectionTable* cross_section_tables,
    HistoryPoint* history,
    std::uint64_t history_capacity,
    unsigned long long* history_count,
    unsigned int* history_overflow,
    FinalNeutron* final_neutrons,
    std::uint32_t final_capacity,
    unsigned int* final_count,
    GenerationNeutron* generation,
    std::uint64_t generation_capacity,
    unsigned long long* generation_count,
    unsigned int* generation_overflow,
    unsigned long long* next_track_id,
    unsigned long long* next_queue_count,
    const float* species_abundances,
    unsigned long long* counters,
    unsigned int* collision_limit_hit,
    unsigned int* no_cross_section_hit,
    unsigned int* track_overflow) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= current_count) {
        return;
    }

    Track track = current_queue[index];
    curandStatePhilox4_32_10_t state = random_states[track.id];
    std::uint32_t collision_count = 0;
    appendHistory(history, history_capacity, history_count, history_overflow,
                  track.id, collision_count, track.energy_ev);

    while (collision_count < max_collisions_per_track) {
        float xs[kSpeciesCount][kChannelCount]{};
        float species_total[kSpeciesCount]{};
        for (int species = 0; species < kSpeciesCount; ++species) {
            for (int channel = 0; channel < kChannelCount; ++channel) {
                xs[species][channel] = interpolateCrossSection(
                    track.energy_ev,
                    cross_section_tables[species * kChannelCount + channel]);
                species_total[species] += xs[species][channel];
            }
        }

        float total_weight = 0.0f;
        for (int species = 0; species < kSpeciesCount; ++species) {
            total_weight += species_total[species] * species_abundances[species];
        }
        if (!(total_weight > 0.0f) || !isfinite(total_weight)) {
            atomicAdd(no_cross_section_hit, 1U);
            break;
        }

        const double uniform =
            static_cast<double>(curand(&state)) * (1.0 / 4294967296.0);
        const double species_choice = uniform * total_weight;
        double cumulative_weight = 0.0;
        int species = kSpeciesCount - 1;
        for (int candidate = 0; candidate < kSpeciesCount; ++candidate) {
            cumulative_weight +=
                species_total[candidate] * species_abundances[candidate];
            if (species_choice < cumulative_weight) {
                species = candidate;
                break;
            }
        }
        const int target_mass_number = speciesMassNumber(species);
        const float species_xs[kChannelCount] = {
            xs[species][Elastic], xs[species][Inelastic],
            xs[species][Capture], xs[species][Fission]};
        const double channel_choice =
            static_cast<double>(curand(&state)) * (1.0 / 4294967296.0) *
            species_total[species];
        double cumulative = species_xs[Elastic];
        int channel = Elastic;
        if (channel_choice >= cumulative) {
            cumulative += species_xs[Inelastic];
            channel = Inelastic;
            if (channel_choice >= cumulative) {
                cumulative += species_xs[Capture];
                channel = Capture;
                if (channel_choice >= cumulative) {
                    channel = Fission;
                }
            }
        }

        ++collision_count;
        atomicAdd(&counters[species * kCountersPerSpecies + channel], 1ULL);
        if (channel == Fission) {
            const int region_counter =
                track.energy_ev < 1.0f ? ThermalRegion
                : track.energy_ev < 1.0e6f ? ResonanceRegion
                                           : FastRegion;
            atomicAdd(
                &counters[species * kCountersPerSpecies + kChannelCount + region_counter],
                1ULL);
        }

        if (channel == Elastic) {
            const float mass = static_cast<float>(target_mass_number);
            if (target_mass_number == 1) {
                track.energy_ev = clampEnergy(track.energy_ev * 1.0e-6f);
            } else {
                const float ratio = (mass - 1.0f) / (mass + 1.0f);
                track.energy_ev = clampEnergy(track.energy_ev * ratio * ratio);
            }
            appendHistory(history, history_capacity, history_count, history_overflow,
                          track.id, collision_count, track.energy_ev);
            continue;
        }
        if (channel == Inelastic) {
            track.energy_ev = clampEnergy(track.energy_ev * 0.5f);
            appendHistory(history, history_capacity, history_count, history_overflow,
                          track.id, collision_count, track.energy_ev);
            continue;
        }

        appendHistory(history, history_capacity, history_count, history_overflow,
                      track.id, collision_count, track.energy_ev);
        if (track.primary) {
            const unsigned int slot = atomicAdd(final_count, 1U);
            if (slot < final_capacity) {
                final_neutrons[slot] = {
                    track.id, collision_count, track.energy_ev,
                    static_cast<std::uint8_t>(channel)};
            } else {
                atomicAdd(track_overflow, 1U);
            }
        }

        if (channel == Capture) {
            break;
        }

        const std::uint32_t multiplicity =
            fissionMultiplicity(state, target_mass_number);
        for (std::uint32_t child = 0; child < multiplicity; ++child) {
            const unsigned long long child_id = atomicAdd(next_track_id, 1ULL);
            if (child_id >= track_capacity || child_id > 0xffffffffULL) {
                atomicAdd(track_overflow, 1U);
                continue;
            }

            curandStatePhilox4_32_10_t child_state;
            curand_init(seed, child_id, 0, &child_state);
            const float child_energy = sampleBirthEnergyEv(child_state);
            random_states[child_id] = child_state;

            if (track.energy_ev >= 1.0f) {
                const unsigned long long slot = atomicAdd(next_queue_count, 1ULL);
                if (slot < track_capacity) {
                    next_queue[slot] = {
                        child_energy, static_cast<std::uint32_t>(child_id), 0};
                } else {
                    atomicAdd(track_overflow, 1U);
                }
            } else {
                appendHistory(history, history_capacity, history_count, history_overflow,
                              static_cast<std::uint32_t>(child_id), 0, child_energy);
                const unsigned long long slot = atomicAdd(generation_count, 1ULL);
                if (slot < generation_capacity) {
                    generation[slot] = {
                        static_cast<std::uint32_t>(child_id), child_energy};
                } else {
                    atomicAdd(generation_overflow, 1U);
                }
            }
        }
        random_states[track.id] = state;
        return;
    }

    random_states[track.id] = state;
    if (collision_count >= max_collisions_per_track) {
        atomicAdd(collision_limit_hit, 1U);
    }
}

template <typename T>
T readValue(std::ifstream& input, const std::string& description) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) {
        throw std::runtime_error("Could not read " + description);
    }
    return value;
}

std::vector<CrossSectionPoint> readTable(
    std::ifstream& input,
    int expected_mass_number,
    int expected_mt,
    bool required) {
    const std::uint32_t mass_number =
        readValue<std::uint32_t>(input, "cross-section isotope mass number");
    const std::uint32_t mt = readValue<std::uint32_t>(input, "cross-section MT");
    const std::uint32_t count = readValue<std::uint32_t>(input, "cross-section point count");
    if (mass_number != static_cast<std::uint32_t>(expected_mass_number) ||
        mt != static_cast<std::uint32_t>(expected_mt) ||
        (count == 0 && required) || (count > 0 && count < 2)) {
        throw std::runtime_error("Unexpected or incomplete cross-section table");
    }

    std::vector<CrossSectionPoint> points(count);
    if (count > 0) {
        input.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(points.size() * sizeof(CrossSectionPoint)));
        if (!input) {
            throw std::runtime_error("Could not read cross-section points");
        }
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        if (!std::isfinite(point.energy_ev) || !std::isfinite(point.cross_section_b) ||
            point.energy_ev <= 0.0f || point.cross_section_b <= 0.0f ||
            (i > 0 && point.energy_ev < points[i - 1].energy_ev)) {
            throw std::runtime_error("Invalid U-238 cross-section table values");
        }
    }
    return points;
}

void writeBinary(const std::filesystem::path& path, const void* data, std::size_t size) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not create output file: " + path.string());
    }
    output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!output) {
        throw std::runtime_error("Could not write output file: " + path.string());
    }
}

std::uint64_t parseUnsigned(const char* text, const char* label) {
    std::size_t consumed = 0;
    const std::string value(text);
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error(std::string("Invalid ") + label);
    }
    return parsed;
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

}  // namespace

int main(int argc, char** argv) {
#if defined(D2O_URANIUM) || defined(H2O_URANIUM)
    constexpr int expected_argc = 10;
#else
    constexpr int expected_argc = 8;
#endif
    if (argc != expected_argc) {
#if defined(D2O_URANIUM)
        std::cerr << "Usage: d2o_natural_uranium_monte_carlo <xs.bin> <output-dir> "
                     "<initial-count> <max-tracks> <max-history-points> <seed> "
                     "<max-collisions> <water-molecules-per-uranium-atom> "
                     "<U-238-to-U-235 ratio>\n";
#elif defined(H2O_URANIUM)
        std::cerr << "Usage: h2o_uranium_monte_carlo <xs.bin> <output-dir> "
                     "<initial-count> <max-tracks> <max-history-points> <seed> "
                     "<max-collisions> <water-molecules-per-uranium-atom> "
                     "<U-238-to-U-235 ratio>\n";
#else
        std::cerr << "Usage: u238_monte_carlo <xs.bin> <output-dir> <initial-count> "
                     "<max-tracks> <max-history-points> <seed> <max-collisions>\n";
#endif
        return 2;
    }

    std::vector<void*> allocations;
    try {
        const auto native_start = std::chrono::steady_clock::now();
        const std::filesystem::path xs_path(argv[1]);
        const std::filesystem::path output_dir(argv[2]);
        const std::uint64_t initial_count_64 = parseUnsigned(argv[3], "initial count");
        const std::uint64_t max_tracks_64 = parseUnsigned(argv[4], "track capacity");
        const std::uint64_t max_history_points = parseUnsigned(argv[5], "history capacity");
        const std::uint64_t seed = parseUnsigned(argv[6], "random seed");
        const std::uint64_t max_collisions_64 = parseUnsigned(argv[7], "collision limit");
#if defined(D2O_URANIUM) || defined(H2O_URANIUM)
        const std::uint64_t water_molecules =
            parseUnsigned(argv[8], "water molecules per uranium atom");
        const std::uint64_t uranium_ratio =
            parseUnsigned(argv[9], "U-238 to U-235 ratio");
        if (water_molecules == 0 || (uranium_ratio != 138 && uranium_ratio != 49)) {
            throw std::runtime_error(
                "Water molecule count must be positive and uranium ratio must be 1:138 or 1:49");
        }
#endif
        if (initial_count_64 == 0 || initial_count_64 > max_tracks_64 ||
            max_tracks_64 > std::numeric_limits<std::uint32_t>::max() ||
            max_history_points < initial_count_64 ||
            max_collisions_64 == 0 ||
            max_collisions_64 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Simulation sizes are invalid or exceed supported limits");
        }

        const auto initial_count = static_cast<std::uint32_t>(initial_count_64);
        const auto track_capacity = static_cast<std::uint32_t>(max_tracks_64);
        const auto max_collisions = static_cast<std::uint32_t>(max_collisions_64);

        std::ifstream input(xs_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Could not open cross-section file: " + xs_path.string());
        }
        constexpr int mt_order[kChannelCount] = {2, 4, 102, 18};
        std::array<float, kSpeciesCount> host_species_abundances{};
#if defined(D2O_URANIUM)
        host_species_abundances = {
            static_cast<float>(2.0 * water_molecules),
            static_cast<float>(water_molecules),
            1.0f / static_cast<float>(uranium_ratio + 1),
            static_cast<float>(uranium_ratio) /
                static_cast<float>(uranium_ratio + 1)};
#elif defined(H2O_URANIUM)
        host_species_abundances = {
            static_cast<float>(2.0 * water_molecules),
            static_cast<float>(water_molecules),
            1.0f / static_cast<float>(uranium_ratio + 1),
            static_cast<float>(uranium_ratio) /
                static_cast<float>(uranium_ratio + 1)};
#elif defined(NATURAL_URANIUM)
        host_species_abundances = {1.0f, 138.0f};
#else
        host_species_abundances = {1.0f};
#endif
        std::array<std::vector<CrossSectionPoint>,
                   kSpeciesCount * kChannelCount> host_tables{};
        for (int species = 0; species < kSpeciesCount; ++species) {
            const int mass_number = speciesMassNumber(species);
            for (int channel = 0; channel < kChannelCount; ++channel) {
                host_tables[species * kChannelCount + channel] =
                    readTable(input, mass_number, mt_order[channel],
                              mass_number >= 235);
            }
        }
        if (input.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Unexpected trailing bytes in cross-section file");
        }

        std::array<DeviceCrossSectionTable, kSpeciesCount * kChannelCount>
            host_device_tables{};
        for (int table = 0; table < kSpeciesCount * kChannelCount; ++table) {
            CrossSectionPoint* device_points = nullptr;
            const auto bytes = host_tables[table].size() * sizeof(CrossSectionPoint);
            if (bytes > 0) {
                checkCuda(cudaMalloc(&device_points, bytes), "cudaMalloc cross-section table");
                allocations.push_back(device_points);
                checkCuda(cudaMemcpy(device_points, host_tables[table].data(), bytes,
                                     cudaMemcpyHostToDevice),
                          "copy cross-section table");
            }
            host_device_tables[table] = {
                device_points, static_cast<std::uint32_t>(host_tables[table].size())};
        }
        DeviceCrossSectionTable* device_tables = nullptr;
        checkCuda(cudaMalloc(&device_tables, sizeof(host_device_tables)),
                  "cudaMalloc cross-section descriptors");
        allocations.push_back(device_tables);
        checkCuda(cudaMemcpy(device_tables, host_device_tables.data(),
                             sizeof(host_device_tables), cudaMemcpyHostToDevice),
                  "copy cross-section descriptors");

        Track* queues[2]{};
        curandStatePhilox4_32_10_t* random_states = nullptr;
        HistoryPoint* history = nullptr;
        FinalNeutron* final_neutrons = nullptr;
        GenerationNeutron* generation = nullptr;
        unsigned long long* counters = nullptr;
        unsigned long long* history_count = nullptr;
        unsigned int* history_overflow = nullptr;
        unsigned int* final_count = nullptr;
        unsigned long long* generation_count = nullptr;
        unsigned int* generation_overflow = nullptr;
        unsigned long long* next_track_id = nullptr;
        unsigned long long* next_queue_count = nullptr;
        float* device_species_abundances = nullptr;
        unsigned int* collision_limit_hit = nullptr;
        unsigned int* no_cross_section_hit = nullptr;
        unsigned int* track_overflow = nullptr;
        auto allocate = [&](auto** pointer, std::size_t count, const char* description) {
            using Element = std::remove_pointer_t<std::remove_pointer_t<decltype(pointer)>>;
            checkCuda(cudaMalloc(pointer, count * sizeof(Element)), description);
            allocations.push_back(*pointer);
        };
        allocate(&queues[0], track_capacity, "cudaMalloc queue A");
        allocate(&queues[1], track_capacity, "cudaMalloc queue B");
        allocate(&random_states, track_capacity, "cudaMalloc random states");
        allocate(&history, static_cast<std::size_t>(max_history_points),
                 "cudaMalloc history points");
        allocate(&final_neutrons, initial_count, "cudaMalloc final set");
        allocate(&generation, track_capacity, "cudaMalloc new generation set");
        allocate(&counters, kCounterCount, "cudaMalloc collision counters");
        allocate(&history_count, 1, "cudaMalloc history count");
        allocate(&history_overflow, 1, "cudaMalloc history overflow");
        allocate(&final_count, 1, "cudaMalloc final count");
        allocate(&generation_count, 1, "cudaMalloc generation count");
        allocate(&generation_overflow, 1, "cudaMalloc generation overflow");
        allocate(&next_track_id, 1, "cudaMalloc total track count");
        allocate(&next_queue_count, 1, "cudaMalloc next queue count");
        allocate(&device_species_abundances, kSpeciesCount,
                 "cudaMalloc species abundances");
        checkCuda(cudaMemcpy(device_species_abundances,
                             host_species_abundances.data(),
                             sizeof(host_species_abundances), cudaMemcpyHostToDevice),
                  "copy species abundances");
        allocate(&collision_limit_hit, 1, "cudaMalloc collision-limit status");
        allocate(&no_cross_section_hit, 1, "cudaMalloc missing-cross-section status");
        allocate(&track_overflow, 1, "cudaMalloc track overflow");

        checkCuda(cudaMemset(counters, 0, kCounterCount * sizeof(*counters)), "clear counters");
        checkCuda(cudaMemset(history_count, 0, sizeof(*history_count)), "clear history count");
        checkCuda(cudaMemset(history_overflow, 0, sizeof(*history_overflow)),
                  "clear history overflow");
        checkCuda(cudaMemset(final_count, 0, sizeof(*final_count)), "clear final count");
        checkCuda(cudaMemset(generation_count, 0, sizeof(*generation_count)),
                  "clear generation count");
        checkCuda(cudaMemset(generation_overflow, 0, sizeof(*generation_overflow)),
                  "clear generation overflow");
        checkCuda(cudaMemcpy(next_track_id, &initial_count_64, sizeof(initial_count_64),
                             cudaMemcpyHostToDevice),
                  "initialize total track count");
        checkCuda(cudaMemset(history_overflow, 0, sizeof(*history_overflow)),
                  "clear history overflow");
        checkCuda(cudaMemset(collision_limit_hit, 0, sizeof(*collision_limit_hit)),
                  "clear collision-limit status");
        checkCuda(cudaMemset(no_cross_section_hit, 0, sizeof(*no_cross_section_hit)),
                  "clear missing-cross-section status");
        checkCuda(cudaMemset(track_overflow, 0, sizeof(*track_overflow)),
                  "clear track overflow");

        constexpr int threads = 256;
        const int initial_blocks =
            static_cast<int>((initial_count + threads - 1) / threads);
        const auto initialization_start = std::chrono::steady_clock::now();
        initializePopulation<<<initial_blocks, threads>>>(
            queues[0], random_states, initial_count, seed);
        checkCuda(cudaGetLastError(), "launch initialization kernel");
        checkCuda(cudaDeviceSynchronize(), "initialize neutron population");
        const double initialization_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - initialization_start).count();

        std::uint32_t current_count = initial_count;
        int current_queue_index = 0;
        std::uint64_t wave = 0;
        const auto transport_start = std::chrono::steady_clock::now();
        while (current_count > 0) {
            const int next_queue_index = 1 - current_queue_index;
            checkCuda(cudaMemset(next_queue_count, 0, sizeof(*next_queue_count)),
                      "clear next queue count");
            const int blocks = static_cast<int>((current_count + threads - 1) / threads);
            simulatePopulation<<<blocks, threads>>>(
                queues[current_queue_index], current_count, queues[next_queue_index],
                track_capacity, random_states, seed, max_collisions,
                device_tables,
                history, max_history_points, history_count, history_overflow,
                final_neutrons, initial_count, final_count, generation, track_capacity,
                generation_count, generation_overflow, next_track_id, next_queue_count,
                device_species_abundances, counters, collision_limit_hit,
                no_cross_section_hit, track_overflow);
            checkCuda(cudaGetLastError(), "launch neutron transport kernel");
            checkCuda(cudaDeviceSynchronize(), "simulate neutron population");

            unsigned int collision_limit = 0;
            unsigned int missing_xs = 0;
            unsigned int track_errors = 0;
            unsigned int history_errors = 0;
            unsigned int generation_errors = 0;
            unsigned long long next_count = 0;
            checkCuda(cudaMemcpy(&collision_limit, collision_limit_hit, sizeof(collision_limit),
                                 cudaMemcpyDeviceToHost), "read collision-limit status");
            checkCuda(cudaMemcpy(&missing_xs, no_cross_section_hit, sizeof(missing_xs),
                                 cudaMemcpyDeviceToHost), "read cross-section status");
            checkCuda(cudaMemcpy(&track_errors, track_overflow, sizeof(track_errors),
                                 cudaMemcpyDeviceToHost), "read track status");
            checkCuda(cudaMemcpy(&history_errors, history_overflow, sizeof(history_errors),
                                 cudaMemcpyDeviceToHost), "read history status");
            checkCuda(cudaMemcpy(&generation_errors, generation_overflow,
                                 sizeof(generation_errors), cudaMemcpyDeviceToHost),
                      "read generation status");
            checkCuda(cudaMemcpy(&next_count, next_queue_count, sizeof(next_count),
                                 cudaMemcpyDeviceToHost), "read next population size");
            if (collision_limit || missing_xs || track_errors || history_errors ||
                generation_errors || next_count > track_capacity) {
                std::cerr << "Simulation stopped: collision-limit tracks="
                          << collision_limit << ", missing-cross-section tracks="
                          << missing_xs << ", track overflows=" << track_errors
                          << ", history overflows=" << history_errors
                          << ", generation overflows=" << generation_errors
                          << ", queued tracks=" << next_count << '\n';
                throw std::runtime_error(
                    "Simulation stopped at a configured resource limit or missing "
                    "cross section; no complete result will be reported");
            }
            current_count = static_cast<std::uint32_t>(next_count);
            current_queue_index = next_queue_index;
            ++wave;
            if (wave > track_capacity) {
                throw std::runtime_error("Simulation exceeded the maximum fission generation depth");
            }
        }
        const double transport_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - transport_start).count();

        unsigned long long history_size = 0;
        unsigned int final_size = 0;
        unsigned long long generation_size = 0;
        unsigned long long total_tracks = 0;
        std::array<unsigned long long, kCounterCount> host_counters{};
        checkCuda(cudaMemcpy(&history_size, history_count, sizeof(history_size),
                             cudaMemcpyDeviceToHost), "read history size");
        checkCuda(cudaMemcpy(&final_size, final_count, sizeof(final_size),
                             cudaMemcpyDeviceToHost), "read final size");
        checkCuda(cudaMemcpy(&generation_size, generation_count, sizeof(generation_size),
                             cudaMemcpyDeviceToHost), "read new generation size");
        checkCuda(cudaMemcpy(&total_tracks, next_track_id, sizeof(total_tracks),
                             cudaMemcpyDeviceToHost), "read total track count");
        checkCuda(cudaMemcpy(host_counters.data(), counters, sizeof(host_counters),
                             cudaMemcpyDeviceToHost), "read collision counters");
        if (history_size > max_history_points || final_size > initial_count ||
            generation_size > track_capacity) {
            throw std::runtime_error("Simulation output exceeded its allocated capacity");
        }

        std::vector<HistoryPoint> host_history(static_cast<std::size_t>(history_size));
        std::vector<FinalNeutron> host_final(final_size);
        std::vector<GenerationNeutron> host_generation(
            static_cast<std::size_t>(generation_size));
        const auto output_start = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(host_history.data(), history, host_history.size() * sizeof(HistoryPoint),
                             cudaMemcpyDeviceToHost), "copy history output");
        checkCuda(cudaMemcpy(host_final.data(), final_neutrons,
                             host_final.size() * sizeof(FinalNeutron), cudaMemcpyDeviceToHost),
                  "copy final neutron set");
        checkCuda(cudaMemcpy(host_generation.data(), generation,
                             host_generation.size() * sizeof(GenerationNeutron),
                             cudaMemcpyDeviceToHost), "copy new generation set");

        std::filesystem::create_directories(output_dir);
        writeBinary(output_dir / "history_points.bin", host_history.data(),
                    host_history.size() * sizeof(HistoryPoint));
        std::ofstream finals(output_dir / "final_neutrons.csv");
        finals << "track_id,collision_count,energy_ev,terminal_event\n";
        for (const auto& item : host_final) {
            finals << item.track_id << ',' << item.collision_count << ','
                   << item.energy_ev << ','
                   << (item.terminal_channel == Capture ? "capture" : "fission") << '\n';
        }
        std::ofstream next_generation(output_dir / "new_generation.csv");
        next_generation << "track_id,birth_energy_ev\n";
        for (const auto& item : host_generation) {
            next_generation << item.track_id << ',' << item.energy_ev << '\n';
        }
        std::ofstream summary(output_dir / "summary.txt");
        summary << "Initial neutrons: " << initial_count << '\n'
                << "Total neutron tracks including fission offspring: " << total_tracks << '\n'
                ;
        for (int species = 0; species < kSpeciesCount; ++species) {
            const char* isotope_name = speciesName(species);
            summary << "Collision count, " << isotope_name << " elastic: "
                    << host_counters[species * kCountersPerSpecies + Elastic] << '\n'
                    << "Collision count, " << isotope_name << " inelastic: "
                    << host_counters[species * kCountersPerSpecies + Inelastic] << '\n'
                    << "Collision count, " << isotope_name << " capture: "
                    << host_counters[species * kCountersPerSpecies + Capture] << '\n'
                    << "Collision count, " << isotope_name << " fission: "
                    << host_counters[species * kCountersPerSpecies + Fission] << '\n'
                    << isotope_name << " fission events in thermal region (E < 1 eV): "
                    << host_counters[species * kCountersPerSpecies + kChannelCount +
                                     ThermalRegion] << '\n'
                    << isotope_name
                    << " fission events in resonance region (1 <= E < 1e6 eV): "
                    << host_counters[species * kCountersPerSpecies + kChannelCount +
                                     ResonanceRegion] << '\n'
                    << isotope_name << " fission events in fast region (E >= 1e6 eV): "
                    << host_counters[species * kCountersPerSpecies + kChannelCount +
                                     FastRegion] << '\n';
        }
        summary << "Final set (initial neutrons terminated by capture/fission): "
                << final_size << '\n'
                << "New-generation set size (requested k-infinity count): "
                << generation_size << '\n'
                << "Recorded history points for all neutron tracks: " << history_size << '\n'
                << "Fission offspring transport waves: " << wave << '\n'
                << "CUDA initialization kernel time (ms): " << initialization_ms << '\n'
                << "CUDA transport kernels and synchronization time (ms): "
                << transport_ms << '\n'
                << "Host result export time (ms): "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - output_start).count() << '\n'
                << "Native executable elapsed time (ms): "
                << std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - native_start).count() << '\n';
        if (!finals || !next_generation || !summary) {
            throw std::runtime_error("Could not write one or more simulation output files");
        }

        std::cout << "Simulation complete.\n"
                  << "Initial neutrons: " << initial_count << '\n'
                  << "Total neutron tracks: " << total_tracks << '\n'
                  ;
        for (int species = 0; species < kSpeciesCount; ++species) {
            const char* isotope_name = speciesName(species);
            std::cout << isotope_name << " collisions (el/inl/capture/fission): "
                      << host_counters[species * kCountersPerSpecies + Elastic] << '/'
                      << host_counters[species * kCountersPerSpecies + Inelastic] << '/'
                      << host_counters[species * kCountersPerSpecies + Capture] << '/'
                      << host_counters[species * kCountersPerSpecies + Fission] << '\n'
                      << isotope_name << " fission (thermal/resonance/fast): "
                      << host_counters[species * kCountersPerSpecies + kChannelCount +
                                       ThermalRegion] << '/'
                      << host_counters[species * kCountersPerSpecies + kChannelCount +
                                       ResonanceRegion] << '/'
                      << host_counters[species * kCountersPerSpecies + kChannelCount +
                                       FastRegion] << '\n';
        }
        std::cout
                  << "Final primary neutrons: " << final_size << '\n'
                  << "New generation set / requested k-infinity count: "
                  << generation_size << '\n'
                  << "History points: " << history_size << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Simulation error: " << error.what() << '\n';
        for (void* allocation : allocations) {
            cudaFree(allocation);
        }
        return 1;
    }

    for (void* allocation : allocations) {
        cudaFree(allocation);
    }
    return 0;
}
