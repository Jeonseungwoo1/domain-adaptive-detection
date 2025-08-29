import os, json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import shutil

CLASS_NAMES = ['bicycle', 'bird', 'car', 'cat', 'dog', 'person']
CLASS2ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}  # COCO category_id starts at 1
CATEGORY_ID_TO_INDEX = {i +1: i for i in range(len(CLASS_NAMES))}

def convert_voc_to_coco(image_ids, image_dir, anno_dir):
    images, annotations = [], []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)]
    ann_id = 1

    for img_id, file_id in enumerate(tqdm(image_ids)):
        img_file = f"{file_id}.jpg"
        xml_file = os.path.join(anno_dir, f"{file_id}.xml")
        img_path = os.path.join(image_dir, img_file)

        width, height = Image.open(img_path).size

        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        root = ET.parse(xml_file).getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in CLASS2ID:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS2ID[name],
                "bbox": [xmin, ymin, w, h],  # COCO format
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
def copy_images(image_ids, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for image_id in image_ids:
        src_path = os.path.join(src_dir, f"{image_id}.jpg")
        dst_path = os.path.join(dst_dir, f"{image_id}.jpg")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)



def run_conversion(dataset_root, name):
    for split in ['train', 'test']:
        list_path = os.path.join(dataset_root, 'ImageSets', 'Main', f'{split}.txt')
        with open(list_path) as f:
            image_ids = [line.strip() for line in f.readlines()]

        coco = convert_voc_to_coco(
            image_ids,
            os.path.join(dataset_root, 'JPEGImages'),
            os.path.join(dataset_root, 'Annotations')
        )

        ann_dir = os.path.join(dataset_root, 'annotations')
        os.makedirs(ann_dir, exist_ok=True)
        coco_json_path = os.path.join(ann_dir, f'{name}_{split}.json')
        with open(coco_json_path, 'w') as f:
            json.dump(coco, f, indent=2)
        
        copy_images(image_ids, os.path.join(dataset_root, 'JPEGImages'), os.path.join(dataset_root, 'images', split))

if __name__ == "__main__":
    names = ['clipart', 'watercolor', 'comic']

    for name in names:
        run_conversion(f'./data/{name}', name)
