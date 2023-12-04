# @article{akyon2022sahi,
#   title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
#   author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
#   journal={2022 IEEE International Conference on Image Processing (ICIP)},
#   doi={10.1109/ICIP46576.2022.9897990},
#   pages={966-970},
#   year={2022}
# }

# When running this program, run in the command line and specify the following: 

# image_dir = Dataset_Full
# dataset_json_path = inside foldder of function
# output_dir = Dataset_Tiled

import fire
from sahi.scripts.slice_coco import slice
from tqdm import tqdm

SLICE_SIZE_LIST = [640] ## For ease of use with other scripts, please only add one slice size to the list
OVERLAP_RATIO_LIST = [0] ## Please add only one overlap ratio to the list
IGNORE_NEGATIVE_SAMPLES = False


def slice_visdrone(image_dir: str, dataset_json_path: str, output_dir: str):
    total_run = len(SLICE_SIZE_LIST) * len(OVERLAP_RATIO_LIST)
    current_run = 1
    for slice_size in SLICE_SIZE_LIST:
        for overlap_ratio in OVERLAP_RATIO_LIST:
            tqdm.write(
                f"{current_run} of {total_run}: slicing for slice_size={slice_size}, overlap_ratio={overlap_ratio}"
            )
            slice(
                image_dir=image_dir,
                dataset_json_path=dataset_json_path,
                output_dir=output_dir,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            current_run += 1


if __name__ == "__main__":
    fire.Fire(slice_visdrone)