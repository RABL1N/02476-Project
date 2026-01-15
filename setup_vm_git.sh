#!/bin/bash

# Configuration
VM_INSTANCE="instance-20260113-110032"
VM_ZONE="europe-west1-d"
VM_USER="rasmusbernthlinnemann_gmail_com"
REMOTE_DIR="~/mlops_project"
GITHUB_REPO="https://github.com/RABL1N/02476-Project.git"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up git repository on VM...${NC}"

# Check if git is installed on VM
echo -e "${YELLOW}Checking if git is installed on VM...${NC}"
gcloud compute ssh --zone="$VM_ZONE" "${VM_USER}@${VM_INSTANCE}" --command="which git || sudo apt-get update && sudo apt-get install -y git"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to install git${NC}"
    exit 1
fi

# Clone or update the repository
echo -e "${YELLOW}Setting up repository at $REMOTE_DIR...${NC}"
gcloud compute ssh --zone="$VM_ZONE" "${VM_USER}@${VM_INSTANCE}" --command="
    if [ -d $REMOTE_DIR/.git ]; then
        echo 'Repository already exists, pulling latest changes...'
        cd $REMOTE_DIR
        git pull origin main || git pull origin master
    else
        echo 'Cloning repository from GitHub...'
        rm -rf $REMOTE_DIR
        git clone $GITHUB_REPO $REMOTE_DIR
    fi
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to set up repository${NC}"
    exit 1
fi

# Install uv if not already installed
echo -e "${YELLOW}Checking if uv is installed...${NC}"
gcloud compute ssh --zone="$VM_ZONE" "${VM_USER}@${VM_INSTANCE}" --command="
    if ! command -v uv &> /dev/null; then
        echo 'Installing uv...'
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo 'Adding uv to PATH in shell profile...'
        if [ -f ~/.bashrc ] && ! grep -q '\$HOME/.local/bin' ~/.bashrc; then
            echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc
        fi
        if [ -f ~/.profile ] && ! grep -q '\$HOME/.local/bin' ~/.profile; then
            echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.profile
        fi
        export PATH=\"\$HOME/.local/bin:\$PATH\"
    else
        echo 'uv is already installed'
    fi
"

echo -e "${GREEN}Repository setup complete!${NC}"
echo -e "${GREEN}Project is now available at: $REMOTE_DIR on ${VM_INSTANCE}${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. SSH into the VM: gcloud compute ssh ${VM_INSTANCE} --zone=${VM_ZONE}"
echo "2. Navigate to project: cd $REMOTE_DIR"
echo "3. If uv command not found, run: source ~/.bashrc (or restart your shell)"
echo "4. Install dependencies: uv sync"
echo ""
echo -e "${YELLOW}To sync changes in the future, run:${NC}"
echo "  ./sync_vm.sh"
