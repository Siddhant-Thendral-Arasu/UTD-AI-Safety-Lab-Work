import os
import tarfile

source_folder = "/data/home/dal667613/task_4_data"
destination_folder = "/data/home/dal667613/NEW_extracted_data"

os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith(".gz"):
        tar_path = os.path.join(source_folder, filename)
        print(f"Extracting {tar_path}...")

        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        extract_path = os.path.join(destination_folder, base_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_path)
            print(f"Done extracting to: {extract_path}")
        except Exception as e:
            print(f"Error extracting {filename}: {e}")