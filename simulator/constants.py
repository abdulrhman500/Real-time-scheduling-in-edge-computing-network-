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



# server_prototypes = [
    
#     {
#         "type": "midserver",
#         "version": 1,
#         "name": "Standard Edge Server",
#         "freq": 15e9,  # 15 GHz
#         "energy_per_cycle": 1.5e-9,
#         "uplink_bandwidth":15, #bits Per sec
#         "downlink_bandwidth":15 #bits Per sec
        
        
#     },
#     {
#         "type": "highserver",
#         "version": 1,
#         "name": "High-Performance Edge Server",
#         "freq": 30e9,  # 30 GHz
#         "energy_per_cycle": 1e-9,  # More efficient energy consumption
#         "uplink_bandwidth":15, #bits Per sec
#         "downlink_bandwidth":15 #bits Per sec
        
#     }
# ]

server_prototypes = [
    {
        "server_id": 1,
        "gateway": "Gateway High",
        "freq": 3.2e9,  # 3.2 GHz
        "TDP": 125,  # 125 Watts
        "IPC": 1.5,  # 1.5 Instructions per cycle
        "uplink_bandwidth": 2e9,  # 2 Gbps
        "downlink_bandwidth": 2e9,  # 2 Gbps
        "uplink_cost": 0.005,  # $0.005 per bit
        "downlink_cost": 0.005,  # $0.005 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 32,  # 32 cores
        "energy_unit_price": 0.07  # $0.07 per unit of energy
    },
    {
        "server_id": 2,
        "gateway": "Gateway Standard",
        "freq": 2.5e9,  # 2.5 GHz
        "TDP": 85,  # 85 Watts
        "IPC": 1.2,  # 1.2 Instructions per cycle
        "uplink_bandwidth": 1e9,  # 1 Gbps
        "downlink_bandwidth": 1e9,  # 1 Gbps
        "uplink_cost": 0.01,  # $0.01 per bit
        "downlink_cost": 0.01,  # $0.01 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 16,  # 16 cores
        "energy_unit_price": 0.10  # $0.10 per unit of energy
    },
    {
        "server_id": 3,
        "gateway": "Gateway Real-Time",
        "freq": 2.2e9,  # 2.2 GHz
        "TDP": 95,  # 95 Watts
        "IPC": 1.0,  # 1.0 Instructions per cycle
        "uplink_bandwidth": 800e6,  # 800 Mbps
        "downlink_bandwidth": 800e6,  # 800 Mbps
        "uplink_cost": 0.015,  # $0.015 per bit
        "downlink_cost": 0.015,  # $0.015 per bit
        "instruction_size": 64,  # 64-bit instructions
        "num_cores": 8,  # 8 cores
        "energy_unit_price": 0.12  # $0.12 per unit of energy
    }
]
