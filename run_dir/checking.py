import subprocess

def submit_job(script_path):
    # The --parsable flag makes sbatch return ONLY the job ID
    command = ["sbatch", "--parsable", script_path]

    try:
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # result.stdout will be the Job ID (e.g., "12345\n")
        job_id = result.stdout.strip()

        print(f"Success! Captured Job ID: {job_id}")
        return job_id

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        return None

# Usage
my_job_id = submit_job("my_script.sh")