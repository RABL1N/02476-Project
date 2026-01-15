#!/bin/bash

# Configuration
VM_INSTANCE="instance-20260113-110032"
VM_ZONE="europe-west1-d"
VM_USER="rasmusbernthlinnemann_gmail_com"
REMOTE_DIR="~/mlops_project"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Syncing VM with latest changes from GitHub...${NC}"

# Pull latest changes from GitHub
gcloud compute ssh --zone="$VM_ZONE" "${VM_USER}@${VM_INSTANCE}" --command="
    if [ ! -d $REMOTE_DIR/.git ]; then
        echo 'Error: Repository not found. Run ./setup_vm_git.sh first.'
        exit 1
    fi
    
    cd $REMOTE_DIR
    echo 'Fetching latest changes...'
    git fetch origin
    
    echo 'Checking current branch...'
    CURRENT_BRANCH=\$(git rev-parse --abbrev-ref HEAD)
    echo \"Current branch: \$CURRENT_BRANCH\"
    
    echo 'Pulling latest changes...'
    git pull origin \$CURRENT_BRANCH || git pull origin main || git pull origin master
    
    if [ \$? -eq 0 ]; then
        echo 'Git sync complete!'
        echo ''
        echo 'Recent commits:'
        git log --oneline -5
    else
        echo 'Warning: There may be local changes or conflicts. Check the repository manually.'
        exit 1
    fi
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to sync repository${NC}"
    exit 1
fi

# Pull latest data using DVC
echo -e "${YELLOW}Pulling latest data with DVC...${NC}"
gcloud compute ssh --zone="$VM_ZONE" "${VM_USER}@${VM_INSTANCE}" --command="
    cd $REMOTE_DIR
    
    # Ensure uv is in PATH
    export PATH=\"\$HOME/.local/bin:\$PATH\"
    
    if ! command -v uv &> /dev/null; then
        echo 'Error: uv is not installed. Run ./setup_vm_git.sh first.'
        exit 1
    fi
    
    echo 'Running: uv run dvc pull'
    uv run dvc pull
    
    DVC_EXIT_CODE=\$?
    if [ \$DVC_EXIT_CODE -eq 0 ]; then
        echo 'DVC pull complete!'
    else
        echo ''
        echo 'WARNING: DVC pull failed. This is likely a permissions issue.'
        echo 'The VM service account needs access to the GCS bucket.'
        echo ''
        echo 'To fix this, run from your local machine:'
        echo '  ./setup_vm_gcs_permissions.sh'
        echo ''
        echo 'Or authenticate with user credentials on the VM:'
        echo '  gcloud auth application-default login'
        echo ''
        exit \$DVC_EXIT_CODE
    fi
"

DVC_SYNC_EXIT=$?
if [ $DVC_SYNC_EXIT -ne 0 ]; then
    echo -e "${RED}Error: Failed to pull data with DVC${NC}"
    echo -e "${YELLOW}See the error message above for troubleshooting steps.${NC}"
    exit $DVC_SYNC_EXIT
fi

echo -e "${GREEN}VM synced successfully with code and data!${NC}"
