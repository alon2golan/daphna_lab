# Daphna Lab Onboarding

## Remote Connect
The connection is made via SSH. There are several computers in the lab that you can connect to:
- kami.cs.huji.ac.il
- inari.cs.huji.ac.il
- takara.cs.huji.ac.il
- hachiman.cs.huji.ac.il

In addition, there is access to the university's general virtual machine:
- phoenix.cs.huji.ac.il 
- phoenix-gw.cs.huji.ac.il 

## SSH
How to connect remotely to the lab computers using SSH:
There are two options:

### Jump host:
```
ssh -J <user>@bava.cs.huji.ac.il <user>@<machine>   
```
Example:
```
ssh -J alongol@bava.cs.huji.ac.il alongol@takara.cs.huji.ac.il
```

### Rumba VPN:
First connect to the university VPN using the link https://rumba.huji.ac.il/. Then ssh directly to the machine:
```
ssh <user>@<machine> 
```
Full explanation is here: https://ca.huji.ac.il/Rumba

## Storage
Most of your work will be done inside your lab folder:
```
/cs/labs/daphna/<user>
```

You also have your user folder (but it's pretty quota limited):
```
/cs/usr/<user>
```

## Repository and Installation 

### Clone the repo
First, you need to clone the lab repo:
```
cd /cs/labs/daphna/<user>/
mkdir projects
cd projects/
git clone https://github.com/alon2golan/daphna_lab.git
```

### Create a Python Virtual Environment
Next you create a python virtual env for this repo. Most of the users in the lab use python3.9. As far as I checked it works also with python3.11 so I recommend to use python 3.11:

```
cd /cs/labs/daphna/<user>/
mkdir envs/
cd envs/
virtualenv -p python3.11 venv311
cd /cs/labs/daphna/<user>/projects/daphna_lab/
ln -s /cs/labs/daphna/<user>/envs/venv311/ venv # create a symbolic link to the venv inside the project
```

### Install Requirements
Now you have to install the requirements inside the virtual env:
```
cd /cs/labs/daphna/<user>/projects/daphna_lab/
source venv/bin/activate
pip install -r deep-al/requirements.txt
```

### Create an Alias (Optional)
You can add an alias to your `.zshrc` / `.zsh_aliases` file to activate the environment:

```bash
activenv() {
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null) || {
        echo "Not inside a Git repository."
        return 1
    }

   for env_dir in .venv venv .env env; do
        if [ -f "$root/$env_dir/bin/activate" ]; then
            echo "✅ Activating environment: $root/$env_dir"
            source "$root/$env_dir/bin/activate"
            return
        fi
    done
    echo "No virtual environment found in $root"
}
```

It should work wherever you are inside the repository. Just type:
```
activenv
```

### Run an Example
Now you can check that everything installed and works fine. Run an example:
```
cd /cs/labs/daphna/<user>/projects/daphna_lab/deep-al/tools/
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name auto --initial_size 0 --budget 100
```

## VSCode Setup

### 1. Prerequisites
- VS Code installed on your local machine
- Remote server(s) with SSH access
- Your username and hostname/IP of the remote server
- Optional: SSH key setup (recommended for passwordless login)

### 2. Install the Remote - SSH Extension
1. Open VS Code
2. Go to the Extensions sidebar (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for Remote - SSH (by Microsoft)
4. Click Install

This extension allows VS Code to connect to remote servers over SSH.

### 3. Configure SSH Hosts
We'll set up your SSH hosts using the `~/.ssh/config` file. This allows using jump hosts and simplifies connections.

Open your SSH config file:
```
code ~/.ssh/config
```

In Windows:
```
C:\Users\<local_user>\.ssh\config
```

Or open with any text editor.

Add the jump host (example: bava.cs.huji.ac.il):
```
Host bava
    HostName bava.cs.huji.ac.il
    User your_username
    IdentityFile ~/.ssh/id_rsa  # optional if you use SSH keys
```

Add the target host using the jump host:
```
Host takara
    HostName takara.cs.huji.ac.il
    User your_username
    ProxyJump bava
    IdentityFile ~/.ssh/id_rsa  # optional
```

✅ Explanation:
- `Host` is the alias you'll use in VS Code
- `HostName` is the real server address
- `User` is your SSH username
- `ProxyJump` tells SSH to first connect to the jump host

### 4. Connect to Remote Host in VS Code
1. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Search for: `Remote-SSH: Connect to Host...`
3. Select your target host (takara)
4. VS Code will open a new window connected to the remote host. It may install a small VS Code server on the remote machine the first time.

### 5. Open Your Remote Project
In the remote VS Code window, go to:
1. File → Open Folder
2. Browse to the folder containing your project (e.g., `/cs/labs/daphna/alongol/projects/daphna_lab`)
3. Click OK

✅ You can now edit, run, and debug code directly on the remote machine.

### 6. Using Python Virtual Environments
If your project has a virtual environment (e.g., `.venv` or `venv`):
1. Open the terminal in VS Code (remote terminal)
2. Activate the environment:
```
source venv/bin/activate
```
3. Select the interpreter in VS Code (Python extension):
   - Press Ctrl+Shift+P → Python: Select Interpreter
   - Choose venv/bin/python

Now VS Code will use your virtual environment for running scripts, debugging, and IntelliSense.

### 7. Tips for Jump Hosts
You can test your SSH config in a terminal:
```
ssh takara
```
It should automatically connect via bava.

For multiple jump hosts:
```
ProxyJump host1,host2
```

If you use a custom SSH key or port, specify in the Host block:
```
IdentityFile ~/.ssh/my_key
Port 2222
```

### 8. Optional: Automating Virtual Environment Activation
You can create a small helper function in `.bashrc` or `.zshrc`:
```bash
activenv() {
    root=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "Not in a Git repo"; return 1; }
    for env_dir in .venv venv env; do
        if [ -f "$root/$env_dir/bin/activate" ]; then
            source "$root/$env_dir/bin/activate"
            return
        fi
    done
    echo "No virtual environment found in $root"
}
```

Then simply run:
```
activenv
```
inside your project to automatically activate the venv.

## Cluster Usage
Needs to be done from a virtual machine `phoenix-gw.cs.huji.ac.il`.

### Prepare a Script File
First need to prepare a script file.

Example: simple hello world script at `/cs/labs/daphna/alongol/projects/daphna_lab/onboarding/sbatch_print_arguments.sh`:

```bash
#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 2
#SBATCH --time=2:00:00
#SBATCH --array=1-3
#SBATCH --output=example_print_arguments_%j.out

dir=/cs/labs/daphna/$USER/projects/daphna_lab/
cd $dir
source $dir/venv/bin/activate
cd $dir/onboarding/
python example_print_arguments.py $SLURM_ARRAY_TASK_ID
```

This script runs a very simple script which prints the input:

```python
import sys
import time

def main():
    print(f"input is: {sys.argv[1:]}")
    time.sleep(30)

if __name__ == "__main__":
    main()
```

### Run it with sbatch
```
ssh phoenix-gw
cd /cs/labs/daphna/$USER/
mkdir test_sbatch/
cd test_sbatch/
sbatch /cs/labs/daphna/$USER/projects/daphna_lab/onboarding/sbatch_print_arguments.sh
```

### Track your queue
To track your queue in sbatch:
```
squeue -u $USER
```

Stdout/stderr of the runnings can be found in the current dir:
```
/cs/labs/daphna/$USER/test_sbatch/example_print_arguments_*.out
```

### Running with GPU
Now a bit more complicated - let's run on sbatch `train_al.py` with GPU:
```
sbatch /cs/labs/daphna/$USER/projects/daphna_lab/onboarding/sbatch_gpu_example.sh
```
