import numpy as np
import random
import matplotlib.pyplot as plt

class Task:
    def __init__(self, task_id, arrival_time, deadline, computation_workload, data_volume, current_cloudlet):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.deadline = deadline
        self.computation_workload = computation_workload
        self.data_volume = data_volume
        self.current_cloudlet = current_cloudlet
        self.assigned_cloudlet = None
        self.start_time = None
        self.end_time = None

class Cloudlet:
    def __init__(self, cloudlet_id, capacity, upload_bandwidth, download_bandwidth, energy_cost, network_cost_in, network_cost_out):
        self.cloudlet_id = cloudlet_id
        self.capacity = capacity
        self.upload_bandwidth = upload_bandwidth
        self.download_bandwidth = download_bandwidth
        self.energy_cost = energy_cost
        self.network_cost_in = network_cost_in
        self.network_cost_out = network_cost_out
        self.tasks = []

def cloudlet_selection(tasks, cloudlets):
    for task in tasks:
        min_cost = float('inf')
        selected_cloudlet = None
        for cloudlet in cloudlets:
            data_transfer_time = task.data_volume / min(cloudlet.upload_bandwidth, cloudlet.download_bandwidth)
            computation_time = task.computation_workload / cloudlet.capacity
            total_time = data_transfer_time + computation_time
            
            if total_time <= (task.deadline - task.arrival_time):
                transfer_cost = task.data_volume * (cloudlet.network_cost_out + cloudlet.network_cost_in)
                computation_cost = task.computation_workload * cloudlet.energy_cost
                total_cost = transfer_cost + computation_cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    selected_cloudlet = cloudlet

        if selected_cloudlet:
            selected_cloudlet.tasks.append(task)
            task.assigned_cloudlet = selected_cloudlet.cloudlet_id
        else:
            print(f"Task {task.task_id} cannot be assigned to any cloudlet.")

def reverse_task_scheduling(cloudlet):
    cloudlet.tasks.sort(key=lambda x: x.deadline, reverse=True)
    current_time = 0
    for task in cloudlet.tasks:
        task.start_time = max(current_time, task.arrival_time)
        task.end_time = task.start_time + (task.computation_workload / cloudlet.capacity)
        current_time = task.end_time

def normalize_task_sizes(tasks, cloudlets):
    for task in tasks:
        if task.assigned_cloudlet is not None:
            assigned_cloudlet = next(c for c in cloudlets if c.cloudlet_id == task.assigned_cloudlet)
            task.normalized_data_size = task.data_volume / min(assigned_cloudlet.upload_bandwidth, assigned_cloudlet.download_bandwidth)

def admission_control(cloudlets):
    for cloudlet in cloudlets:
        while True:
            if not cloudlet.tasks:
                break
            critical_interval = max(cloudlet.tasks, key=lambda x: x.data_volume / (x.deadline - x.arrival_time))
            if sum(task.normalized_data_size for task in cloudlet.tasks if task.deadline <= critical_interval.deadline) > 1:
                cloudlet.tasks.remove(critical_interval)
            else:
                break

def mcf_edf_scheduling(cloudlet):
    cloudlet.tasks.sort(key=lambda x: x.deadline)
    for task in cloudlet.tasks:
        task.start_time = max(task.arrival_time, (sum(t.data_volume for t in cloudlet.tasks if t.deadline <= task.deadline) / cloudlet.upload_bandwidth))
        task.end_time = task.start_time + (task.data_volume / cloudlet.download_bandwidth)

def closest_cloudlet_selection(tasks, cloudlets):
    for task in tasks:
        selected_cloudlet = None
        min_distance = float('inf')
        for cloudlet in cloudlets:
            distance = abs(int(cloudlet.cloudlet_id[1:]) - int(task.current_cloudlet[1:]))
            if distance < min_distance:
                min_distance = distance
                selected_cloudlet = cloudlet
        
        if selected_cloudlet:
            selected_cloudlet.tasks.append(task)
            task.assigned_cloudlet = selected_cloudlet.cloudlet_id
        else:
            print(f"Task {task.task_id} cannot be assigned to any cloudlet.")

def fcfs_edf_scheduling(cloudlet):
    cloudlet.tasks.sort(key=lambda x: (x.arrival_time, x.deadline))
    current_time = 0
    for task in cloudlet.tasks:
        task.start_time = max(current_time, task.arrival_time)
        task.end_time = task.start_time + (task.computation_workload / cloudlet.capacity)
        current_time = task.end_time

def generate_cloudlets(num_cloudlets):
    cloudlets = []
    for i in range(num_cloudlets):
        cloudlet_id = f'C{i+1}'
        capacity = random.uniform(10, 20)  # Uniform distribution for capacity
        upload_bandwidth = random.uniform(10, 20)  # Uniform distribution for upload bandwidth
        download_bandwidth = random.uniform(10, 20)  # Uniform distribution for download bandwidth
        energy_cost = abs(random.gauss(0.5, 0.1))  # Normal distribution for energy cost
        network_cost_in = abs(random.gauss(0.1, 0.05))  # Normal distribution for network cost in
        network_cost_out = abs(random.gauss(0.1, 0.05))  # Normal distribution for network cost out
        cloudlets.append(Cloudlet(cloudlet_id, capacity, upload_bandwidth, download_bandwidth, energy_cost, network_cost_in, network_cost_out))
    return cloudlets

def generate_tasks(num_tasks, cloudlets):
    tasks = []
    for i in range(num_tasks):
        task_id = f'T{i+1}'
        arrival_time = random.uniform(0, 10)  # Uniform distribution for arrival time
        deadline = arrival_time + random.uniform(5, 15)  # Uniform distribution for deadline
        computation_workload = abs(random.gauss(10, 3))  # Normal distribution for computation workload
        data_volume = abs(random.gauss(100, 30))  # Normal distribution for data volume
        current_cloudlet = random.choice(cloudlets).cloudlet_id  # Assign to a random current cloudlet
        tasks.append(Task(task_id, arrival_time, deadline, computation_workload, data_volume, current_cloudlet))
    return tasks

def collect_results(tasks):
    admitted_tasks = [task for task in tasks if task.assigned_cloudlet is not None]
    admission_rate = len(admitted_tasks) / len(tasks)
    average_cost = np.mean([task.computation_workload for task in admitted_tasks]) if admitted_tasks else 0
    return admission_rate, average_cost

def plot_results(results, baseline_results):
    scales, admission_rates, average_costs = zip(*results)
    _, baseline_admission_rates, baseline_average_costs = zip(*baseline_results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(scales, admission_rates, marker='o', label='Proposed Algorithm')
    ax1.plot(scales, baseline_admission_rates, marker='x', label='Baseline Algorithm')
    ax1.set_title('Task Admission Rate')
    ax1.set_xlabel('Number of Cloudlets')
    ax1.set_ylabel('Admission Rate')
    ax1.legend()
    
    ax2.plot(scales, average_costs, marker='o', label='Proposed Algorithm')
    ax2.plot(scales, baseline_average_costs, marker='x', label='Baseline Algorithm')
    ax2.set_title('Average Per-Task Cost')
    ax2.set_xlabel('Number of Cloudlets')
    ax2.set_ylabel('Average Cost')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    scales = [20, 50, 70,100 ,1000]
    results = []
    baseline_results = []
    
    for num_cloudlets in scales:
        cloudlets = generate_cloudlets(num_cloudlets)
        tasks = generate_tasks(50, cloudlets)
        
        # Proposed Algorithm
        cloudlet_selection(tasks, cloudlets)
        for cloudlet in cloudlets:
            reverse_task_scheduling(cloudlet)
        normalize_task_sizes(tasks, cloudlets)
        admission_control(cloudlets)
        for cloudlet in cloudlets:
            mcf_edf_scheduling(cloudlet)
        admission_rate, average_cost = collect_results(tasks)
        results.append((num_cloudlets, admission_rate, average_cost))
        
        # Baseline Algorithm
        cloudlets_baseline = generate_cloudlets(num_cloudlets)
        tasks_baseline = generate_tasks(50, cloudlets_baseline)
        closest_cloudlet_selection(tasks_baseline, cloudlets_baseline)
        for cloudlet in cloudlets_baseline:
            fcfs_edf_scheduling(cloudlet)
        baseline_admission_rate, baseline_average_cost = collect_results(tasks_baseline)
        baseline_results.append((num_cloudlets, baseline_admission_rate, baseline_average_cost))
    
    # Plot results
    plot_results(results, baseline_results)

if __name__ == "__main__":
    main()

print("This script is a simplified version to illustrate the methodology. For real-world applications, additional details and optimizations would be necessary, especially considering more complex network topologies, varying task priorities, and more advanced cost functions.")
