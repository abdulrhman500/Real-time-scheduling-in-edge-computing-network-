from components.Task import Task

class Server:
    # Thermal Design Power (TDP)
    def __init__(self, server_id, gateway, freq, TDP, IPC, uplink_bandwidth, downlink_bandwidth, uplink_cost, downlink_cost, instruction_size, num_cores, energy_unit_price):
        self.id = server_id
        self.gateway = gateway
        self.freq = freq  # Number of CPU cycles per second (frequency)
        self.TDP = TDP  # Thermal Design Power
        self.IPC = IPC  # Instructions per cycle
        self.IPS = self.IPC * self.freq  # Instructions per second
        self.EPI = self.TDP / self.IPS  # Energy per instruction
        self.energy_unit_price = energy_unit_price  # Cost per unit of energy
        self.instruction_size = instruction_size  # Instruction size
        self.uplink_bandwidth = uplink_bandwidth  # Uplink bandwidth (bits per second)
        self.downlink_bandwidth = downlink_bandwidth  # Downlink bandwidth (bits per second)
        self.uplink_cost = uplink_cost  # Uplink cost per bit
        self.downlink_cost = downlink_cost  # Downlink cost per bit
        self.num_cores = num_cores  # Number of CPU cores
        self.tasks_running = []
        self.is_space_shared = True  # Whether the server uses space-sharing for tasks

    def processing_cost(self, task: Task):
        cost = self.EPI * task.num_instruction_to_be_executed
        return cost

    def processing_time(self, task: Task):
        return task.num_instruction_to_be_executed / self.IPS

    def remaining_comp_capacity(self, start_time, end_time):
        if self.is_space_shared:
            used_cores = [0] * self.num_cores
            for task in self.tasks_running:
                task_end_time = task.starting_time + self.processing_time(task)
                if task.starting_time < end_time and task_end_time > start_time:
                    overlap_start = max(task.starting_time, start_time)
                    overlap_end = min(task_end_time, end_time)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        for i in range(self.num_cores):
                            if used_cores[i] == 0:
                                used_cores[i] = 1
                                break
            remaining_cores = self.num_cores - sum(used_cores)
            return remaining_cores * self.IPS / self.num_cores
        else:
            used_capacity = 0
            for task in self.tasks_running:
                task_end_time = task.starting_time + self.processing_time(task)
                if task.starting_time < end_time and task_end_time > start_time:
                    overlap_start = max(task.starting_time, start_time)
                    overlap_end = min(task_end_time, end_time)
                    overlap_duration = overlap_end - overlap_start
                    used_capacity += (overlap_duration / (end_time - start_time)) * self.IPS
            remaining_capacity = self.IPS - used_capacity
            return remaining_capacity

    def schedule(self, tasks):
        reverse_tasks = []
        for task in tasks:
            reverse_tasks.append(task)
        reverse_tasks.sort(key=lambda t: t.deadline)
        scheduled_tasks = []
        while reverse_tasks:
            next_task = reverse_tasks.pop(0)
            task_end_time = next_task.deadline
            task_start_time = task_end_time - self.processing_time(next_task)
            next_task.starting_time = task_start_time
            next_task.end_time = task_end_time
            scheduled_tasks.append(next_task)
            self.tasks_running.append(next_task)
            next_task.execution_server = self
        return scheduled_tasks

    def remove_task(self, task_id):
        for task in self.tasks_running:
            if task.id == task_id:
                self.tasks_running.remove(task)
                break

    def __str__(self):
        return (f"\nServer_ID: {self.id}\n"
                f"Gateway: {self.gateway}\n"
                f"Frequency (cycles/sec): {self.freq}\n"
                f"TDP: {self.TDP}\n"
                f"IPC: {self.IPC}\n"
                f"IPS: {self.IPS}\n"
                f"EPI (energy/instruction): {self.EPI}\n"
                f"Energy Unit Price: {self.energy_unit_price}\n"
                f"Instruction Size: {self.instruction_size}\n"
                f"Uplink Bandwidth: {self.uplink_bandwidth}\n"
                f"Downlink Bandwidth: {self.downlink_bandwidth}\n"
                f"Uplink Cost/bit: {self.uplink_cost}\n"
                f"Downlink Cost/bit: {self.downlink_cost}\n"
                f"Number of Cores: {self.num_cores}\n"
                f"Tasks Running: {self.tasks_running}\n" 
                f"Is Space Shared: {self.is_space_shared}\n"
        )