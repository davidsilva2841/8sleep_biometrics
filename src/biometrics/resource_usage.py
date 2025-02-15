import os


def get_memory_usage_unix():
    pid = os.getpid()  # Current process ID
    page_size = os.sysconf('SC_PAGE_SIZE')    # Page size in bytes
    rss = int(open(f'/proc/{pid}/statm').read().split()[1])  # Resident Set Size in pages
    memory_usage_mb = (rss * page_size) / (1024 ** 2)        # Convert to MB
    return memory_usage_mb


def get_free_memory_mb():
    """
    Returns the free memory in MB by reading /proc/meminfo (Linux only).
    """
    meminfo_path = '/proc/meminfo'
    if not os.path.exists(meminfo_path):
        raise EnvironmentError("This function is supported only on Linux systems with /proc/meminfo.")

    with open(meminfo_path, 'r') as meminfo:
        for line in meminfo:
            if line.startswith('MemFree:'):
                # Extract the value in KB
                free_kb = int(line.split()[1])
                # Convert to MB
                free_mb = free_kb / 1024
                return round(free_mb, 2)

    # If MemFree is not found
    return 0.0


def get_available_memory_mb():
    """
    Returns the available memory in MB by reading /proc/meminfo (Linux only).
    """
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        raise EnvironmentError("This function is supported only on Linux systems with /proc/meminfo.")

    with open(meminfo_path, "r") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])  # Extract value in KB
                return available_kb / 1024

    return 0.0
