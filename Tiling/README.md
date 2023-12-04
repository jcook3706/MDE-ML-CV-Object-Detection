# Tiling Visdrone Dataset

This set of files allows the users to tile their dataset using the SAHI libraries for training. 
Each python script has a specific use, and in order to get the full results, it is recommended to
use the files in this order: 

1. vis2coco.py
2. slicing_test.py
3. convert2yolo.py

## vis2coco.py

This function will convert the visdrone annotations into the coco format from the visdrone format, 
which will allow the user to use the SAHI slice functions called in later files

## slicing_test.py

This file slices the dataset into the size and overlap ratios that the user wants, because of issues
with the json format, we recommend only specifiying one overlap and slice size

## convert2yolo.py

This file contains the functions that will convert the sliced coco dataset annotations into a YOLOV8
compatible format. To use, first edit line 117 of converter.py function inside of the ultralytics package on
your local device from this: 

``` cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class ```

to this:

``` cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] # class ```



Citations for code used: 

> ```
> @article{akyon2022sahi,
>  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
>  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
>  journal={2022 IEEE International Conference on Image Processing (ICIP)},
>  doi={10.1109/ICIP46576.2022.9897990},
>  pages={966-970},
>  year={2022}
>}
> ```
>
> https://github.com/fcakyon/small-object-detection-benchmark/tree/main
