import sys
import platform
import cbor2
from datetime import datetime, timedelta
if platform.system().lower() == 'linux':
    sys.path.append('/home/dac/python_packages/')
    sys.path.append('/home/dac/free-sleep/biometrics/')
    sys.path.append('/home/dac/free-sleep/biometrics/stream/')


import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
import threading

from stream_processor import StreamProcessor
from sleep_detection.cap_data import load_baseline
from load_raw_files import load_piezo_row
from get_logger import get_logger
logger = get_logger()

RIGHT_BASELINE = load_baseline('right')
LEFT_BASELINE = load_baseline('left')

# Global queue for processing decoded biometric data
piezo_record_queue = queue.Queue()


class LatestRawFileHandler(FileSystemEventHandler):
    """Monitors only the latest RAW file and processes CBOR-encoded lines separately."""

    def __init__(self, directory):
        self.directory = directory
        self.latest_file = None
        self.latest_file_obj = None
        self.last_pos = 0  # Track last read position
        self.track_latest_file()

    def track_latest_file(self):
        """Finds the most recent RAW file in the directory and starts tracking it."""
        raw_files = [f for f in os.listdir(self.directory) if f.endswith(".RAW")]
        if not raw_files:
            return

        # Get the latest file by modification time
        raw_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.directory, f)), reverse=True)
        latest_file = os.path.join(self.directory, raw_files[0])

        if latest_file != self.latest_file:
            # If a new file is found, stop tracking the old one
            self.latest_file = latest_file
            logger.debug(f"Now tracking: {self.latest_file}")

            # Close old file handle if open
            if self.latest_file_obj:
                self.latest_file_obj.close()

            # Open the new file for reading in binary mode
            self.latest_file_obj = open(self.latest_file, "rb")
            self.last_pos = 0  # Reset position for new file

    def on_created(self, event):
        """Triggered when a new .RAW file is created."""
        if event.is_directory or not event.src_path.endswith(".RAW"):
            return
        self.track_latest_file()

    def follow_latest_file(self):
        """Reads and decodes new CBOR-encoded lines from the latest file."""
        if not self.latest_file_obj:
            return

        # Move to the last known position before reading
        self.latest_file_obj.seek(self.last_pos)

        while True:
            try:
                # Decode CBOR object from a **single line**
                row = cbor2.load(self.latest_file_obj)  # Load the next CBOR object
                one_minute_ago = datetime.now() - timedelta(minutes=1)

                if 'data' in row:  # Check if 'data' key exists
                    decoded_data = cbor2.loads(row['data'])
                    if decoded_data['type'] != 'piezo-dual':
                        continue
                    record_time = datetime.fromtimestamp(decoded_data['ts'])
                    if one_minute_ago > record_time:
                        continue
                    load_piezo_row(decoded_data, 'right')
                    piezo_record_queue.put(decoded_data)

                # Update last read position
                self.last_pos = self.latest_file_obj.tell()

            except EOFError:
                # No more CBOR objects to read
                break
            except Exception as e:
                logger.error(f"Error decoding CBOR: {e}")
                break


def process_data():
    """
    Consumes from data_queue, maintains a rolling buffer,
    checks presence, calculates heart rate if present.
    """
    piezo_record = piezo_record_queue.get()
    stream_processor = StreamProcessor(piezo_record)
    while True:
        piezo_record = piezo_record_queue.get()
        # Block until new data is available
        if piezo_record is None:
            break

        stream_processor.process_piezo_record(piezo_record)
        piezo_record_queue.task_done()


def watch_directory(directory="/persistent"):
    """Monitors the directory for new RAW files and processes only the latest one."""
    handler = LatestRawFileHandler(directory)
    observer = Observer()
    observer.schedule(handler, directory, recursive=False)
    observer.start()

    # Start biometric processing in a separate thread
    processing_thread = threading.Thread(target=process_data, daemon=True)
    processing_thread.start()

    try:
        while True:
            time.sleep(1)
            handler.track_latest_file()  # Check if a newer file exists
            handler.follow_latest_file()  # Read new CBOR entries line-by-line
    except KeyboardInterrupt:
        observer.stop()
        piezo_record_queue.put(None)  # Send stop signal to processing thread
        processing_thread.join()
    observer.join()

# Start watching and processing the latest .RAW file
watch_directory("/persistent")
