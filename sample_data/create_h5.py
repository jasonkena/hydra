import zipfile
import glob
import h5py
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  #allow large image
from tqdm import tqdm


def convert(input_path, output_path):
    with h5py.File(output_path, 'w') as h5file:
        image_data = []

        png_files = sorted(glob.glob(input_path))
        for png_file in tqdm(png_files, desc="loading pngs from zip"):
            img = Image.open(png_file)
            img_array = np.array(img)

            image_data.append(img_array)

        h5file.create_dataset(
            "main",
            data=np.array(image_data),
            maxshape=(None, *image_data[0].shape),
            chunks=True
        )

    print(f"done, saved to {output_path}")

if __name__ == "__main__":
    convert("sample_data/V1/hydra_export_*.png", "sample_data/V1/EM.h5")
    convert("sample_data/V1/Segmentation_export_*.png", "sample_data/V1/SV.h5")
