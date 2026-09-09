# Connect database on HPC to streetscapes-explorer

This guide is written for the Snellius HPC system, but is largely
reusable for any other system.

The usecase here is to use the streetscapes explorer while the back-end is running
remotely.

You first need to login to snellius, e.g.;

```bash
ssh snellius
```

Next install streetscapes:

```bash
# Install uv if you don't have it yet:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new virtual environment (here named `streetscapes`):
uv venv streetscapes --python 3.14

# Activate the virtual environment:
source streetscapes/bin/activate

# Install streetscapes:
uv pip install streetscapes[sam3,explorer] --pre
```

After this, download images and (optionally) run a segmentation.

To connect to the project data files on HPC, create a new file on snellius called `run_explorer_snellius.sh`:

```bash
#!/bin/bash
#
# Serve the streetscapes explorer on snellius
# usage: sbatch run_explorer_snellius.sh

# SLURM settings
#SBATCH -J streetscapes-explorer
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -p rome
#SBATCH --output=./slurm_%j.out
#SBATCH --error=./slurm_%j.out

# Activate the virtual environment (see setup instructions)
source streetscapes/bin/activate

# Some security: stop script on error and undefined variables
set -euo pipefail

# Choose random port and print instructions to connect
PORT=`shuf -i 5000-5999 -n 1`
LOGIN_HOST_EXT=int5-pub.snellius.surf.nl
LOGIN_HOST_INT=int5

echo "Selected port is: " $PORT
echo
echo "To connect to the notebook type the following command from your local terminal:"
echo "ssh -L ${PORT}:localhost:${PORT} ${USER}@${LOGIN_HOST_EXT}"

ssh -o StrictHostKeyChecking=no -f -N -p 22 -R $PORT:localhost:$PORT $LOGIN_HOST_INT

# Start explorer
streetscapes-explorer --no-open-webpage --port $PORT
```

Wait for the job to be launched, and check the logs for the connection message.
E.g.;

```bash
cat slurm_26376022.out
```

Copy-paste this connection message;

```bash
ssh -L 5882:localhost:5882 ACCOUNT_NAME@int5-pub.snellius.surf.nl
```

And then connect to this server on the [streetscapes exporer](https://urban-m4.github.io/streetscapes-explorer/).
Use the port from the slurm log, the host is localhost (due to the ssh tunnel).
