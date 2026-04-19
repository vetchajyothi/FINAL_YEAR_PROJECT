import os

# 👉 Folder 1 (example: hem scans)
folder1 = "E:\\FINAL_YEAR_PROJECT\\ORIGINAL_DATASET_StrokeType - Copy\\Brain_Stroke_CT-SCAN_image\\val\\hemorrhagic"
prefix1 = "hem_scan_"

# 👉 Folder 2 (example: normal scans)
folder2 = "E:\\FINAL_YEAR_PROJECT\\ORIGINAL_DATASET_StrokeType - Copy\\Brain_Stroke_CT-SCAN_image\\val\\ischaemic"
prefix2 = "isc_scan_"

def rename_images(folder_path, prefix):
    files = sorted(os.listdir(folder_path))
    count = 1

    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            old_path = os.path.join(folder_path, file)

            # keep original extension
            ext = os.path.splitext(file)[1]

            new_name = f"{prefix}_{count}{ext}"
            new_path = os.path.join(folder_path, new_name)

            os.rename(old_path, new_path)
            count += 1

    print(f"Renamed files in {folder_path}")

# 🔁 Run for both folders
rename_images(folder1, prefix1)
rename_images(folder2, prefix2)

print("All folders processed successfully!")