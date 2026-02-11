import argparse
import sys
import time

def parse_args():

    parser = argparse.ArgumentParser(description="Job ID Parser")
    parser.add_argument("--job_id", type=int, required=True, help="Job ID")
    parser.add_argument("--job_name", type=str, required=True, help="Job Name")
	
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    job_id = args.job_id
    job_name = args.job_name
    print(job_id," ",job_name)
    sys.stdout.flush()
    time.sleep(600)