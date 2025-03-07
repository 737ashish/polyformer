import os
import time
import argparse
from datetime import datetime

def monitor_and_transfer(run_dir, output_dir, interval=300):
    """Monitor runs directory and transfer new results periodically"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = os.path.join(output_dir, f"snapshot_{timestamp}")
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Copy latest event files
        os.system(f"cp -r {run_dir}/* {snapshot_dir}/")
        print(f"Snapshot created at: {snapshot_dir}")
        
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="./runs", help="TensorBoard runs directory")
    parser.add_argument("--output_dir", default="./tb_snapshots", help="Output directory for snapshots")
    parser.add_argument("--interval", type=int, default=300, help="Monitoring interval in seconds")
    args = parser.parse_args()
    
    monitor_and_transfer(args.run_dir, args.output_dir, args.interval)
