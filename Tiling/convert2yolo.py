# When running this function, please edit line 117 of converter.py 
# in the convert_coco function in order to preserve the class numbering
# that is found from this: 

# cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class

# to this: 
# cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] # class
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir='Dataset_tiled/sliced/', use_segments=False, use_keypoints=False, cls91to80=False)
