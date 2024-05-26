class Server:
    def __init__(self, server_id, gateway, comp_capacity, energy_per_cycle):
        self.id = server_id
        self.gateway = gateway
        self.comp_capacity = comp_capacity  # Number of CPU cycles per second (frequency)
        self.energy_per_cycle = energy_per_cycle 
