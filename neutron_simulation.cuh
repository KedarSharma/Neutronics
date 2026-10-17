#pragma once

#include <curand_kernel.h>

#include <cstdint>

namespace neutron_sim {

constexpr float kMinimumEnergyEv = 1.0e-5f;
constexpr float kMaximumEnergyEv = 1.0e8f;
constexpr float kFissionSpectrumRatePerMev = 0.776f;

enum class EnergyRegion : std::uint8_t {
    Thermal,
    Resonance,
    Fast,
};

enum class Reaction : std::uint8_t {
    NoInteraction,
    Elastic,
    Inelastic,
    RadiativeCapture,
    Fission,
};

struct Neutron {
    float energy_ev;
    std::uint32_t generation;
    bool active;
};

__global__ void initializeNeutrons(
    Neutron* neutrons,
    curandStatePhilox4_32_10_t* random_states,
    std::uint64_t neutron_count,
    std::uint64_t random_seed);

__global__ void applyNeutronEvents(
    Neutron* neutrons,
    curandStatePhilox4_32_10_t* random_states,
    const Reaction* selected_events,
    const std::uint16_t* target_mass_numbers,
    std::uint64_t neutron_count,
    Neutron* offspring,
    curandStatePhilox4_32_10_t* offspring_random_states,
    std::uint64_t offspring_capacity,
    unsigned long long* offspring_count,
    unsigned long long* offspring_overflow_count,
    unsigned int* invalid_event_count);

__device__ EnergyRegion classifyEnergy(float energy_ev);

}  // namespace neutron_sim
