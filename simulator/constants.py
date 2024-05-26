device_prototypes = [
    {"type": "smartphone", "version": 0, "name": "High-End Smartphone", "comp_capacity": 3.0e9, "energy_per_cycle": 1e-9, "cycle_per_bit": 10, "max_waiting_tasks": 10},
    {"type": "sensor", "version": 0, "name": "Low-Power Sensor Node", "comp_capacity": 0.5e9, "energy_per_cycle": 0.8e-9, "cycle_per_bit": 20, "max_waiting_tasks": 2},
    {"type": "sbc", "version": 0, "name": "Single-Board Computer", "comp_capacity": 1.4e9, "energy_per_cycle": 1.1e-9, "cycle_per_bit": 14, "max_waiting_tasks": 5},
    {"type": "wearable", "version": 0, "name": "Wearable Device", "comp_capacity": 1.2e9, "energy_per_cycle": 1.3e-9, "cycle_per_bit": 16, "max_waiting_tasks": 3}
]
#C = B * log₂(1 + SNR)
# C is the channel capacity (maximum data rate) in bits per second (bps).
# B is the channel bandwidth in Hertz (Hz).
# SNR = Signal-to-Noise Ratio
# The bandwidth of a channel is often directly referred to as its channel width.
standard_gateway = {
    
        "type": "mobile_base_station",
        "version": 1,
        "name": "4G Mobile Base Station",
        "num_channels": 10,
        "channel_bandwidths": [10, 10, 30, 20, 20, 20, 30, 10, 30, 30]  # MHz
}


gateway_prototypes = [
    {
        "type": "residential",
        "version": 1,
        "name": "Basic Home Gateway",
        "num_channels": 3,
        "channel_bandwidths": [5, 10, 15]  # MHz
    },
    {
        "type": "enterprise",
        "version": 2,
        "name": "High-Capacity Enterprise Gateway",
        "num_channels": 8,
        "channel_bandwidths": [20, 20, 20, 20, 30, 30, 40, 50]  # MHz
    },
    {
        "type": "industrial",
        "version": 1,
        "name": "Rugged Industrial Gateway",
        "num_channels": 5,
        "channel_bandwidths": [10, 10, 15, 20, 25]  # MHz
    },
    {
        "type": "mobile",
        "version": 1,
        "name": "Mobile Hotspot",
        "num_channels": 2,
        "channel_bandwidths": [5, 10]  # MHz
    },
    {
        "type": "mobile_base_station",
        "version": 1,
        "name": "4G Mobile Base Station",
        "num_channels": 10,  # More channels for wider coverage
        "channel_bandwidths": [10, 10, 30, 20, 20, 20, 30, 10, 30, 30]  # MHz
    }
]



server_prototypes = [
    
    {
        "type": "midserver",
        "version": 1,
        "name": "Standard Edge Server",
        "comp_capacity": 15e9,  # 15 GHz
        "energy_per_cycle": 1.5e-9,
        
    },
    {
        "type": "highserver",
        "version": 1,
        "name": "High-Performance Edge Server",
        "comp_capacity": 30e9,  # 30 GHz
        "energy_per_cycle": 1e-9,  # More efficient energy consumption
        
    }
]