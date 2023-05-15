import os
import torch
import torchvision.transforms.functional as F
from pycocotools.coco import COCO
from PIL import Image

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, set_name='train', transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, 'raw', f'instances_{self.set_name}2017.json'))
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        pil_img = Image.open(os.path.join(self.root_dir, self.set_name, 'data', path)).convert('RGB')
        target = {}
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = areas
        target['iscrowd'] = iscrowd

        if self.transform is not None:
            img = self.transform(pil_img)

        # Scale the bounding boxes
        x_scale = img.shape[-2] / pil_img.size[0]
        y_scale = img.shape[-1] / pil_img.size[1]
        target['boxes'][:, 0] = target['boxes'][:, 0] * x_scale
        target['boxes'][:, 1] = target['boxes'][:, 1] * y_scale
        target['boxes'][:, 2] = target['boxes'][:, 2] * x_scale
        target['boxes'][:, 3] = target['boxes'][:, 3] * y_scale

        #TODO: Need to define custom collate_fn to process the variable length bounding boxes

        return img, target
    
    def __len__(self):
        return len(self.ids)
