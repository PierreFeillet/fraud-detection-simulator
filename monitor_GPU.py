import subprocess
import time
import re

LOG_FILE = "gpu_monitor.log"

def get_top_gpu_process():
    """Finds the most GPU-intensive application."""
    gpu_process = "Unknown"
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        gpu_processes = [line for line in result.stdout.split("\n") if "gpu" in line.lower()]
        if gpu_processes:
            # Sort by highest CPU usage (approx. indicator of GPU load)
            top_process = sorted(gpu_processes, key=lambda x: float(x.split()[2]), reverse=True)[0]
            gpu_process = top_process.split()[-1]  # Extract process name
    except Exception as e:
        gpu_process = f"Error: {e}"
    return gpu_process

def calculate_severity(gpu_power, gpu_util, gpu_temp):
    """Determines severity level based on GPU power, utilization, and temperature."""
    if gpu_power is None or gpu_util is None or gpu_temp is None:
        return "N/A"

    severity_score = (gpu_power / 20) + (gpu_util / 20) + ((gpu_temp - 50) / 10)

    if severity_score > 10:
        return "CRITICAL 🚨"
    elif severity_score > 7:
        return "HIGH 🔥"
    elif severity_score > 4:
        return "MODERATE ⚠️"
    else:
        return "LOW ✅"

def parse_powermetrics_output(output):
    """Parses powermetrics output for GPU and CPU statistics."""
    gpu_power = None
    gpu_freq = None
    gpu_util = None

    for line in output.split("\n"):
        if "GPU Power:" in line:
            match = re.search(r"GPU Power: (\d+) mW", line)
            if match:
                gpu_power = int(match.group(1)) / 1000  # Convert mW to W

        elif "GPU HW active frequency:" in line:
            match = re.search(r"(\d+) MHz", line)
            if match:
                gpu_freq = int(match.group(1))  # Extract frequency in MHz

        elif "GPU HW active residency:" in line:
            match = re.search(r"(\d+\.\d+)%", line)
            if match:
                gpu_util = float(match.group(1))  # Extract utilization %

    return gpu_power, gpu_freq, gpu_util

def get_istat_temps():
    """Fetches CPU & GPU temperatures using iStats."""
    try:
        result = subprocess.run(["istats"], capture_output=True, text=True)
        cpu_temp = re.search(r"CPU temp:\s+(\d+\.\d+)", result.stdout)
        gpu_temp = re.search(r"GPU temp:\s+(\d+\.\d+)", result.stdout)

        return float(cpu_temp.group(1)) if cpu_temp else None, float(gpu_temp.group(1)) if gpu_temp else None
    except:
        return None, None

def log_data(timestamp, gpu_power, gpu_freq, gpu_util, cpu_temp, gpu_temp, gpu_process, severity):
    """Logs data to a file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp}, {gpu_power}, {gpu_freq}, {gpu_util}, {cpu_temp}, {gpu_temp}, {gpu_process}, {severity}\n")

def run_powermetrics():
    """Runs powermetrics in streaming mode."""
    process = subprocess.Popen(
        ["sudo", "powermetrics", "--samplers", "gpu_power", "-i", "1000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return process

def monitor_gpu(interval=1):
    """Monitors GPU and CPU usage in real-time."""
    print("🔍 Monitoring GPU & CPU Usage... (Press Ctrl+C to stop)\n")
    print(f"{'Time':<10} | {'GPU Power (W)':<15} | {'GPU Freq (MHz)':<15} | {'GPU Util (%)':<12} | {'CPU Temp (°C)':<15} | {'GPU Temp (°C)':<15} | {'Top GPU Process':<20} | {'Severity':<15}")
    print("-" * 130)

    powermetrics_process = run_powermetrics()

    try:
        with open(LOG_FILE, "w") as f:
            f.write("Timestamp, GPU Power (W), GPU Freq (MHz), GPU Util (%), CPU Temp (°C), GPU Temp (°C), Top GPU Process, Severity\n")

        while True:
            line = powermetrics_process.stdout.readline()
            if not line:
                continue  # Skip empty lines

            gpu_power, gpu_freq, gpu_util = parse_powermetrics_output(line)

            # Get CPU & GPU temperatures from iStats
            cpu_temp, gpu_temp = get_istat_temps()

            # Find the top GPU-consuming process
            gpu_process = get_top_gpu_process()

            # Calculate severity level
            severity = calculate_severity(gpu_power, gpu_util, gpu_temp)

            timestamp = time.strftime('%H:%M:%S')
            print(f"{timestamp:<10} | {gpu_power if gpu_power else 'N/A':<15} | {gpu_freq if gpu_freq else 'N/A':<15} | {gpu_util if gpu_util else 'N/A':<12} | {cpu_temp if cpu_temp else 'N/A':<15} | {gpu_temp if gpu_temp else 'N/A':<15} | {gpu_process[:20]:<20} | {severity:<15}")

            log_data(timestamp, gpu_power, gpu_freq, gpu_util, cpu_temp, gpu_temp, gpu_process, severity)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n❌ Stopping GPU Monitoring.")
        powermetrics_process.terminate()

if __name__ == "__main__":
    monitor_gpu()
