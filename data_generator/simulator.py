"""
AI-Powered Connected Vehicle Analytics Platform (BDA)
Continuous Telematics Live Simulator

Continuously produces JSON stream batches into the landing directory to simulate
real-time vehicle telematics pings for Databricks Auto Loader or Structured Streaming.

Usage:
  python data_generator/simulator.py --output-dir data/landing --interval 5 --records-per-batch 20 --total-batches 50
"""

import os
import time
import json
import random
import argparse
import datetime
try:
    from data_generator.generate_telematics import generate_vehicle_master, VehicleState
except ImportError:
    from generate_telematics import generate_vehicle_master, VehicleState

def run_simulator(
    output_dir: str = "data/landing",
    interval: float = 3.0,
    records_per_batch: int = 25,
    num_vehicles: int = 50,
    total_batches: int = 0  # 0 means infinite until Ctrl+C
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" CONNECTED VEHICLE TELEMATICS - LIVE STREAMING SIMULATOR ")
    print("=" * 70)
    print(f"Target landing directory : {output_dir}")
    print(f"Emission interval        : {interval} seconds")
    print(f"Records per batch        : {records_per_batch}")
    print(f"Active simulated fleet   : {num_vehicles} vehicles")
    print(f"Total batches to emit    : {'Infinite (Ctrl+C to stop)' if total_batches == 0 else total_batches}")
    print("=" * 70)

    # Initialize vehicles
    master_df = generate_vehicle_master(num_vehicles)
    master_rows = master_df.to_dict(orient="records")
    fleet = [VehicleState(row) for row in master_rows]

    batch_idx = 1
    try:
        while True:
            batch_records = []
            now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            for _ in range(records_per_batch):
                v = random.choice(fleet)
                record, _ = v.step(delta_seconds=int(interval))
                record["timestamp"] = now_iso
                batch_records.append(record)

            file_name = f"stream_{int(time.time())}_batch_{batch_idx:05d}.json"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w") as f:
                for rec in batch_records:
                    f.write(json.dumps(rec) + "\n")

            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Emitted {file_name} -> {len(batch_records)} records")

            if total_batches > 0 and batch_idx >= total_batches:
                print(f"[+] Reached requested batch count ({total_batches}). Stopping simulator.")
                break

            batch_idx += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[*] Simulator stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Telematics Streaming Simulator")
    parser.add_argument("--output-dir", type=str, default="data/landing", help="Directory where streaming JSON files are deposited")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between batches")
    parser.add_argument("--records-per-batch", type=int, default=25, help="Number of vehicle pings per batch")
    parser.add_argument("--num-vehicles", type=int, default=50, help="Number of vehicles in simulated fleet")
    parser.add_argument("--total-batches", type=int, default=10, help="Total batches to emit (0 for infinite)")
    args = parser.parse_args()

    run_simulator(
        output_dir=args.output_dir,
        interval=args.interval,
        records_per_batch=args.records_per_batch,
        num_vehicles=args.num_vehicles,
        total_batches=args.total_batches
    )
