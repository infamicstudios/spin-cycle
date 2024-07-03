import csv
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import time

def read_valid_freqs(file_path):
    freqs = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            freq, gpu = row
            freq = int(freq.split()[0])  
            if gpu not in freqs:
                freqs[gpu] = []
            freqs[gpu].append(freq)
    return freqs

def set_gpu_frequency(gpu_id, freq):
    subprocess.run(['nvidia-smi', '-i', str(gpu_id), '-lgc', str(freq)], check=True)

def run_spin_cycle():
    result = subprocess.run(['./spin_cycle'], capture_output=True, text=True)
    return result.stdout

def parse_output(output):
    runs = []
    current_run = {}
    for line in output.split('\n'):
        if line.startswith('Device Number:'):
            if current_run:
                runs.append(current_run)
            current_run = {'bins': []}
            current_run['device_id'] = int(line.split(':')[1].strip())
        elif 'Name:' in line:
            current_run['name'] = line.split(':')[1].strip()
        elif '# repetitions' in line:
            current_run['repetitions'] = int(line.split()[-1])
        elif line.startswith('Bin'):
            bin_num, count = map(int, re.findall(r'\d+', line))
            current_run['bins'].append((bin_num, count))
        elif line.startswith('Total measurements:'):
            current_run['total'] = int(line.split(':')[1].strip())
    if current_run:
        runs.append(current_run)
    return runs

def plot_histograms(all_runs):
    devices = set(run['name'] for runs in all_runs.values() for run in runs)
    for device in devices:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(18)
        width = 0.8 / len(all_runs)
        
        for i, (freq, runs) in enumerate(all_runs.items()):
            device_run = next((run for run in runs if run['name'] == device), None)
            if device_run:
                bins = [0] * 18
                for bin_num, count in device_run['bins']:
                    bins[bin_num] = count
                ax.bar(x + i*width, bins, width, label=f'{freq} MHz')
        
        ax.set_xlabel('Bin')
        ax.set_ylabel('Count')
        ax.set_title(f'Histogram for {device}')
        ax.legend()
        ax.set_yscale('symlog')
        plt.tight_layout()
        plt.savefig(f'{device.replace(" ", "_")}_histogram.png')
        plt.close()

def plot_histogram_differences(all_runs):
    devices = set(run['name'] for runs in all_runs.values() for run in runs)
    for device in devices:
        sorted_freqs = sorted(all_runs.keys())
        baseline_freq = sorted_freqs[0]
        
        fig, axs = plt.subplots(len(sorted_freqs)-1, 1, figsize=(15, 8*(len(sorted_freqs)-1)), sharex=True)
        fig.suptitle(f'Histogram of Clock Count Variance for {device}\n(Baseline: {baseline_freq} MHz)')
        
        max_bin = max(max(bin_num for run in runs if run['name'] == device for bin_num, _ in run['bins'])
                      for runs in all_runs.values())
        x = np.arange(max_bin + 1)
        
        baseline_run = next(run for run in all_runs[baseline_freq] if run['name'] == device)
        baseline_bins = dict(baseline_run['bins'])
        
        for i, freq in enumerate(sorted_freqs[1:]):
            ax = axs[i] if len(sorted_freqs) > 2 else axs
            runs = all_runs[freq]
            device_run = next((run for run in runs if run['name'] == device), None)
            if device_run:
                bins = [0] * (max_bin + 1)
                for bin_num, count in device_run['bins']:
                    if bin_num <= max_bin:
                        baseline_count = baseline_bins.get(bin_num, 0)
                        bins[bin_num] = count - baseline_count
                ax.bar(x, bins, width=0.8)
                ax.set_ylabel(f'Count Difference\n({freq} MHz)')
                ax.set_yscale('symlog')
                
        axs[-1].set_xlabel('Bin')
        plt.tight_layout()
        plt.savefig(f'{device.replace(" ", "_")}_difference_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Print debug info
    for freq, runs in all_runs.items():
        print(f"Frequency: {freq} MHz")
        for run in runs:
            print(f"Device: {run['name']}")
            print(f"Total measurements: {run['total']}")
            print("Bin counts:")
            for bin_num, count in run['bins']:
                print(f"  Bin {bin_num}: {count}")
        print("---")

# Main 
valid_freqs = read_valid_freqs('valid_freqs.csv')
all_runs = {}

for gpu_name, freqs in valid_freqs.items():
    for freq in freqs:
        print(f"Setting {gpu_name} to {freq} MHz")
        gpu_id = None
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '-L'], text=True)
        for line in nvidia_smi_output.split('\n'):
            if gpu_name in line:
                gpu_id = line.split(':')[0].split()[-1]
                break
        
        if gpu_id is None:
            print(f"Couldn't find GPU ID for {gpu_name}")
            continue
        
        set_gpu_frequency(gpu_id, freq)
        
        # Wait to ensure freq change
        time.sleep(20)
        
        print("Running spin_cycle...")
        output = run_spin_cycle()
        runs = parse_output(output)
        all_runs[freq] = runs

plot_histograms(all_runs)
plot_histogram_differences(all_runs)
print("Graphs have been saved as PNG files.")

# Reset GPU to default
for gpu_id in range(len(valid_freqs)):
    subprocess.run(['nvidia-smi', '-i', str(gpu_id), '-rgc'], check=True)
