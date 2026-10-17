#include "neutron_simulation.cuh"

#include <cmath>

namespace neutron_sim {
namespace {

__device__ float clampEnergyEv(float energy_ev) {
    return fminf(kMaximumEnergyEv, fmaxf(kMinimumEnergyEv, energy_ev));
}

__device__ float sampleFissionEnergyEv(curandStatePhilox4_32_10_t& state) {
    const float normal_x = curand_normal(&state);
    const float normal_y = curand_normal(&state);
    const float normal_z = curand_normal(&state);
    const float energy_mev =
        (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z) /
        (2.0f * kFissionSpectrumRatePerMev);
    return clampEnergyEv(energy_mev * 1.0e6f);
}

__device__ unsigned int sampleFissionMultiplicity(
    curandStatePhilox4_32_10_t& state,
    std::uint16_t mass_number) {
    const double uniform =
        static_cast<double>(curand(&state)) * (1.0 / 4294967296.0);
    if (mass_number == 235) {
        return static_cast<unsigned int>(2.5 + uniform);
    }
    return static_cast<unsigned int>(2.43 + uniform);
}

__device__ bool isSupportedMassNumber(std::uint16_t mass_number) {
    return mass_number == 1 || mass_number == 2 || mass_number == 16 ||
           mass_number == 235 || mass_number == 238;
}

}  // namespace

__device__ EnergyRegion classifyEnergy(float energy_ev) {
    if (energy_ev < 1.0f) {
        return EnergyRegion::Thermal;
    }
    if (energy_ev < 1.0e6f) {
        return EnergyRegion::Resonance;
    }
    return EnergyRegion::Fast;
}

__global__ void initializeNeutrons(
    Neutron* neutrons,
    curandStatePhilox4_32_10_t* random_states,
    std::uint64_t neutron_count,
    std::uint64_t random_seed) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= neutron_count) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(random_seed, index, 0, &state);
    neutrons[index] = {sampleFissionEnergyEv(state), 0, true};
    random_states[index] = state;
}

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
    unsigned int* invalid_event_count) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= neutron_count || !neutrons[index].active) {
        return;
    }

    Neutron& neutron = neutrons[index];
    curandStatePhilox4_32_10_t state = random_states[index];
    const std::uint16_t mass_number = target_mass_numbers[index];

    switch (selected_events[index]) {
        case Reaction::NoInteraction:
            break;
        case Reaction::Elastic:
            if (!isSupportedMassNumber(mass_number)) {
                atomicAdd(invalid_event_count, 1U);
                break;
            }
            if (mass_number == 1) {
                neutron.energy_ev *= 1.0e-6f;
            } else {
                const float mass = static_cast<float>(mass_number);
                const float ratio = (mass - 1.0f) / (mass + 1.0f);
                neutron.energy_ev *= ratio * ratio;
            }
            neutron.energy_ev = clampEnergyEv(neutron.energy_ev);
            break;
        case Reaction::Inelastic:
            neutron.energy_ev = clampEnergyEv(neutron.energy_ev * 0.5f);
            break;
        case Reaction::RadiativeCapture:
            neutron.active = false;
            break;
        case Reaction::Fission: {
            if (mass_number != 235 && mass_number != 238) {
                atomicAdd(invalid_event_count, 1U);
                break;
            }

            neutron.active = false;
            const unsigned int multiplicity =
                sampleFissionMultiplicity(state, mass_number);
            for (unsigned int child = 0; child < multiplicity; ++child) {
                const unsigned long long slot = atomicAdd(offspring_count, 1ULL);
                if (slot >= offspring_capacity) {
                    atomicAdd(offspring_overflow_count, 1ULL);
                    continue;
                }

                const unsigned long long child_seed =
                    (static_cast<unsigned long long>(curand(&state)) << 32) |
                    curand(&state);
                curandStatePhilox4_32_10_t child_state;
                curand_init(child_seed, slot, 0, &child_state);
                offspring[slot] = {
                    sampleFissionEnergyEv(child_state),
                    neutron.generation + 1,
                    true,
                };
                offspring_random_states[slot] = child_state;
            }
            break;
        }
        default:
            atomicAdd(invalid_event_count, 1U);
            break;
    }

    random_states[index] = state;
}

}  // namespace neutron_sim
