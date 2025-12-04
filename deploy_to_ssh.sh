#!/bin/bash
# Deployment script for running GA simulation on SSH server
# Usage: ./deploy_to_ssh.sh user@hostname [remote_path]

set -e

# Configuration
LOCAL_PROJECT="/Users/hassanshahristani/Documents/IIB/4th_year_project"
REMOTE_USER_HOST="${1:-user@ssh-server}"
REMOTE_PATH="${2:-~/4th_year_project}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GA Simulation Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if [ "$1" == "" ]; then
    echo -e "${YELLOW}Usage: ./deploy_to_ssh.sh user@hostname [remote_path]${NC}"
    echo -e "${YELLOW}Example: ./deploy_to_ssh.sh hs755@gate.eng.cam.ac.uk ~/4th_year_project${NC}"
    echo ""
    echo "This script will:"
    echo "  1. Create a deployment package"
    echo "  2. Transfer files to remote server"
    echo "  3. Set up Python environment on remote"
    echo ""
    exit 1
fi

echo -e "Target: ${YELLOW}${REMOTE_USER_HOST}:${REMOTE_PATH}${NC}"
echo ""

# Step 1: Create deployment directory
echo -e "${GREEN}Step 1: Creating deployment package...${NC}"
DEPLOY_DIR="${LOCAL_PROJECT}/deploy_package"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy essential files
echo "  - Copying tools/"
cp -r "${LOCAL_PROJECT}/tools" "$DEPLOY_DIR/"

echo "  - Copying training/"
cp -r "${LOCAL_PROJECT}/training" "$DEPLOY_DIR/"

echo "  - Copying meshes/"
cp -r "${LOCAL_PROJECT}/meshes" "$DEPLOY_DIR/"

echo "  - Copying Sciurus17 robot model..."
cp -r "${LOCAL_PROJECT}/Sciurus17_mujoco_sim_example" "$DEPLOY_DIR/"
# Remove local venv from robot folder
rm -rf "$DEPLOY_DIR/Sciurus17_mujoco_sim_example/venv"

echo "  - Copying requirements.txt"
cp "${LOCAL_PROJECT}/requirements.txt" "$DEPLOY_DIR/"

# Create necessary directories
mkdir -p "$DEPLOY_DIR/training_results"
mkdir -p "$DEPLOY_DIR/ga_checkpoints"
mkdir -p "$DEPLOY_DIR/ga_stats"
mkdir -p "$DEPLOY_DIR/videos"

# Create setup script for remote server
cat > "$DEPLOY_DIR/setup_remote.sh" << 'SETUP_SCRIPT'
#!/bin/bash
# Remote setup script - run this on the SSH server

set -e

echo "Setting up Python environment..."

# Check Python version
python3 --version

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the GA training:"
echo "  source venv/bin/activate"
echo "  python training/ga_tendon_optimizer_v2_pyramid.py"
echo ""
echo "To run with visualization (requires X11 forwarding):"
echo "  ssh -X user@server  # Connect with X11 forwarding"
echo "  source venv/bin/activate"
echo "  python training/ga_tendon_optimizer_v2_pyramid.py --visualize"
echo ""
echo "To run headless (no display, for training only):"
echo "  export MUJOCO_GL=osmesa  # or egl"
echo "  python training/ga_tendon_optimizer_v2_pyramid.py"
SETUP_SCRIPT

chmod +x "$DEPLOY_DIR/setup_remote.sh"

# Create a run script
cat > "$DEPLOY_DIR/run_training.sh" << 'RUN_SCRIPT'
#!/bin/bash
# Run GA training

source venv/bin/activate

# For headless servers without display
export MUJOCO_GL=osmesa

# Run training
python training/ga_tendon_optimizer_v2_pyramid.py "$@"
RUN_SCRIPT

chmod +x "$DEPLOY_DIR/run_training.sh"

echo -e "${GREEN}Step 2: Transferring files to remote server...${NC}"
echo "  This may take a few minutes..."

# Create remote directory and transfer files
ssh "$REMOTE_USER_HOST" "mkdir -p $REMOTE_PATH"
rsync -avz --progress "$DEPLOY_DIR/" "${REMOTE_USER_HOST}:${REMOTE_PATH}/"

echo ""
echo -e "${GREEN}Step 3: Setting up remote environment...${NC}"
echo "  Running setup script on remote server..."

ssh "$REMOTE_USER_HOST" "cd $REMOTE_PATH && chmod +x setup_remote.sh && ./setup_remote.sh"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To run training on the remote server:"
echo -e "  ${YELLOW}ssh ${REMOTE_USER_HOST}${NC}"
echo -e "  ${YELLOW}cd ${REMOTE_PATH}${NC}"
echo -e "  ${YELLOW}./run_training.sh${NC}"
echo ""
echo "Or run in background with nohup:"
echo -e "  ${YELLOW}nohup ./run_training.sh > training.log 2>&1 &${NC}"
echo ""
echo "To download results:"
echo -e "  ${YELLOW}scp -r ${REMOTE_USER_HOST}:${REMOTE_PATH}/training_results ./results_from_ssh/${NC}"
echo ""

# Cleanup local deploy package
rm -rf "$DEPLOY_DIR"
