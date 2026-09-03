# run on staging node like so:
# salloc -p staging -t 15:00 -n 16
# /home/pkalverla1/Urban-M4/streetscapes/.venv/bin/ipython scripts/count_image_size.py


from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def get_image_size(image_path: Path):
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        # Could log errors if needed
        return None


def process_images(image_paths):
    counter = Counter()
    with Pool(cpu_count()) as pool:
        for size in tqdm(
            pool.imap_unordered(get_image_size, image_paths), total=len(image_paths)
        ):
            if size is not None:
                counter[size] += 1
    return counter


def main():
    image_dir = Path("/projects/prjs0914/streetscapes/data/sources/mapillary/images")
    image_paths = list(image_dir.glob("*.jpeg"))

    print(f"Found {len(image_paths)} images, extracting sizes...")
    counts = process_images(image_paths)

    print("\nImage size counts (width x height):")
    for (w, h), count in counts.most_common():
        print(f"{count:6} images of size {w}x{h}")


if __name__ == "__main__":
    main()
