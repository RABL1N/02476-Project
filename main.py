import shutil
from pathlib import Path

import kagglehub

# Download latest version to cache
cache_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Downloaded to cache: {cache_path}")

# Copy to local repo directory
local_data_dir = Path("data/raw")
local_data_dir.mkdir(parents=True, exist_ok=True)

# Copy the dataset from cache to local directory
cache_path_obj = Path(cache_path)
# The dataset structure has train/test/val directories
# Check if they're directly in cache_path or in a subdirectory
dataset_dirs = ["train", "test", "val"]
source_dir = cache_path_obj

# Check if train/test/val are in a subdirectory (like chest_xray/)
for subdir in cache_path_obj.iterdir():
    if subdir.is_dir() and subdir.name != "__MACOSX":
        subdir_train = subdir / "train"
        if subdir_train.exists():
            source_dir = subdir
            break

# Copy train, test, and val directories
if source_dir.exists():
    copied_any = False
    for dir_name in dataset_dirs:
        src_dir = source_dir / dir_name
        dest_dir = local_data_dir / dir_name
        if src_dir.exists():
            if dest_dir.exists():
                print(f"{dir_name}/ already exists, skipping...")
            else:
                print(f"Copying {dir_name}/...")
                shutil.copytree(src_dir, dest_dir)
                copied_any = True

    if copied_any:
        print(f"\nDataset copied successfully to: {local_data_dir.absolute()}")
    else:
        print(f"\nDataset already exists at: {local_data_dir.absolute()}")
else:
    print(f"Error: Source path {source_dir} does not exist")

print(f"\nLocal dataset path: {local_data_dir.absolute()}")
