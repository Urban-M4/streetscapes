# Streetscapes Explorer

To easy visualizing results from streetscapes (images, segmentations) we developed
the Streetscapes Explorer, a browser based tool.

To run it you will need to have installed streetscapes with the extra `explorer`
dependencies:

```bash
pip install streetscapes==1.0.0a0[explorer]
```

If you have a project set up with some images and segmentations, you start the
explorer "back end" with:

```bash
streetscapes-explorer
```

This will also try to open a webpage. If this fails, visit the webpage manually
at https://urban-m4.github.io/streetscapes-explorer/
When connecting to streetscapes, the webpage will likely ask for permission to
access the local device. This needs to be granted for the explorer to work.

If the explorer does not seem to work, please try a different browser. Firefox and
Chrome have been tested.

For additional options (such as port configuration) see
`streetscapes-explorer --help`.


## Connect Streetscapes Explorer to a project on HPC

This guide is written for the [Snellius HPC system](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660184/Snellius),
but is largely reusable for any other system.

The usecase here is to use the streetscapes explorer while the back-end is running
remotely.

You first need to login to snellius, e.g.;

```bash
ssh -L 5001:localhost:5001 snellius
```

The `-L` option sets up an SSH tunnel from your machine to the login node,
on the specified port.

If you haven't yet, install streetscapes. Note that you only need to do
this the first time. If you already have a streetscapes environment set up,
reuse that one.

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
Once you have some data to visualize, start the explorer with:

```bash
streetscapes-explorer --no-open-webpage
```

And navigate to https://urban-m4.github.io/streetscapes-explorer/?s=http://localhost:5001
The explorer should then display the data of your active streetscapes
project on Snellius.

### Connecting to a compute node

If, for some reason, you need to run the streetscapes explorer backend on a compute
node instead of a login node, create a new file on snellius called `run_explorer_snellius.sh`:

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
#   Note that the `int5` part of the hostname may be outdated at some point
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
