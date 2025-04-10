
import os
import zipfile
import gdown

def download_kitti_sample(output_dir="sample_data"):
    os.makedirs(output_dir, exist_ok=True)
    file_id = "1La3mDJ74E-WNSqKn6d3gPMgQnd-EKhf5"  # Replace with your file ID
    output_path = os.path.join(output_dir, "kitti_sample.zip")

    print("Downloading KITTI sample sequence...")
    gdown.download(id=file_id, output=output_path, quiet=False)

    print("Extracting...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    os.remove(output_path)
    print("Done.")
