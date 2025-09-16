#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Define the image name
IMAGE_NAME="caracal-gpu-docker"

# Get parent directory of your project (where other projects are)
HOST_PROJECTS_DIR=$(dirname "$(pwd)")
CONTAINER_PROJECTS_DIR="/home/appuser/projects"

# Get current user and group IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)

echo "--- Launching JupyterLab container for Caracal development ---"
echo "Access JupyterLab at http://localhost:8888"
echo "Running as user $USER_ID:$GROUP_ID to maintain proper file ownership"
echo "Mounting $HOST_PROJECTS_DIR to $CONTAINER_PROJECTS_DIR"
echo "JupyterLab root: $CONTAINER_PROJECTS_DIR"
echo "Caracal library: $CONTAINER_PROJECTS_DIR/caracal"
echo "To exit, stop the process by pressing Ctrl+C"

# Create user setup command to run inside container
USER_SETUP_CMD="
# Create group if it doesn't exist
if ! getent group $GROUP_ID >/dev/null 2>&1; then
   groupadd -g $GROUP_ID appuser
fi

# Create user if it doesn't exist
if ! id -u appuser >/dev/null 2>&1; then
   useradd -u $USER_ID -g $GROUP_ID -d /home/appuser -m -s /bin/bash appuser
   mkdir -p /home/appuser/.jupyter/runtime
   mkdir -p /home/appuser/.cache/pip
   chown -R appuser:appuser /home/appuser
fi

# Change ownership of projects directory to the user
chown -R appuser:appuser /home/appuser/projects

# Install caracal library from the caracal subdirectory
echo 'Installing Caracal library in editable mode...'
cd /home/appuser/projects/caracal
pip install -e .

# Switch to user and start JupyterLab from projects root
su - appuser -c 'cd /home/appuser/projects && JUPYTER_CONFIG_DIR=/home/appuser/.jupyter JUPYTER_DATA_DIR=/home/appuser/.jupyter JUPYTER_RUNTIME_DIR=/home/appuser/.jupyter/runtime jupyter lab --ip=0.0.0.0 --port=8888 --no-browser'
"

# Run container with the setup command
docker run \
   --gpus all \
   -it \
   --rm \
   -p 8888:8888 \
   -v "$HOST_PROJECTS_DIR":"$CONTAINER_PROJECTS_DIR" \
   "$IMAGE_NAME" \
   bash -c "$USER_SETUP_CMD"
