import os

folder = "downloads_raw"
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print(f"Total files in {folder}: {len(files)}")
