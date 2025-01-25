import paramiko
from scp import SCPClient
import os
from pathlib import Path

from data_manager import DataManager
from toolkit import tools
from config import PROJECT_FOLDER_PATH


PEOPLE = [
    DataManager('david', load=False),
    DataManager('den', load=False),
    DataManager('tally', load=False),
    DataManager('trinity', load=False),
]

def get_ssh_config():
    """Load SSH configuration from ~/.ssh/config."""
    ssh_config_file = Path.home() / ".ssh" / "config"
    ssh_config = paramiko.SSHConfig()
    if ssh_config_file.exists():
        with ssh_config_file.open() as f:
            ssh_config.parse(f)
    return ssh_config


def get_ssh_host_config(ssh_config, host_alias):
    """Retrieve host-specific configuration from SSHConfig."""
    return ssh_config.lookup(host_alias)


def get_file_paths():
    file_paths = []
    for person in PEOPLE:
        file_paths.append(person.piezo_df_file_path)
        file_paths.append(person.heart_rate_file_path)
        file_paths.append(person.breath_rate_file_path)
        file_paths.append(person.hrv_file_path)
        file_paths.append(person.sleep_data_file_path)


    return file_paths


def scp_file_to_remote(local_file, host_alias, remote_path):
    try:
        ssh_config = get_ssh_config()
        host_config = get_ssh_host_config(ssh_config, host_alias)

        if not host_config:
            raise ValueError(f"No SSH configuration found for alias '{host_alias}'")

        # Extract SSH connection details
        hostname = host_config.get("hostname", host_alias)
        username = host_config.get("user", "root")
        port = int(host_config.get("port", 22))
        identity_file = host_config.get("identityfile", [None])[0]

        # Create an SSH client
        ssh = paramiko.SSHClient()

        # Automatically add the server's host key if not already known
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote host
        ssh.connect(
            hostname=hostname,
            username=username,
            port=port,
            key_filename=identity_file
        )

        # Ensure the destination directory exists
        remote_dir = remote_path.rsplit('/', 1)[0]  # Extract directory from the remote path
        stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_dir}')
        stdout.channel.recv_exit_status()  # Wait for command to complete

        # Create an SCP client
        with SCPClient(ssh.get_transport()) as scp:
            # Transfer the file
            scp.put(local_file, remote_path)

        print(f"\nCopied {local_file} successfully copied to {host_alias}:{remote_path}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the SSH connection
        ssh.close()


def copy_to_pod():
    file_paths = get_file_paths()
    bar = tools.progress_bar(file_paths)
    for local_file in file_paths:
        bar.update()
        remote_path = local_file.replace(PROJECT_FOLDER_PATH, '/home/ds/main/8sleep_biometrics/')
        if not os.path.isfile(local_file):
            continue


        scp_file_to_remote(
            local_file=local_file,
            host_alias="desktop",
            remote_path=remote_path
        )


copy_to_pod()
