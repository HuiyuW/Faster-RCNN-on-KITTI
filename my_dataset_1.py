from torch.utils.data import Dataset
import os
import cv2
import torch
import json
from PIL import Image
from lxml import etree


class KITTIDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, kitti_root, transforms=None, txt_name: str = "train.txt"):
        self.root = os.path.join(kitti_root, "training")
        self.img_root = os.path.join(self.root, "image_2")
        self.label_root = os.path.join(self.root, "label_2")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.txt_list = [os.path.join(self.label_root, line.strip() + ".txt")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.txt_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        for txt_path in self.txt_list:
            assert os.path.exists(txt_path), "not found '{}' file.".format(txt_path)

        # read class_indict
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        # read xml
        txt_path = self.txt_list[idx]
        f=open(txt_path)  
        split_lines = f.readlines() 
        #print name
        head, tail = os.path.split(txt_path)
        name= tail[:-4]  
        img_name=name+'.png'   
        img_path=os.path.join('./training/image_2/',img_name) # 路径需要自行修改              
        #print img_path  
        img_size=cv2.imread(img_path).shape  
        image = Image.open(img_path)
        if image.format != "PNG":
            raise ValueError("Image '{}' format not PNG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        for split_line in split_lines:  
            line=split_line.strip().split()

            xmin = int(float(line[4]))
            xmax = int(float(line[6]))
            ymin = int(float(line[5]))
            ymax = int(float(line[7]))

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(txt_path))
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[line[0]])

            iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        txt_path = self.txt_list[idx]
        head, tail = os.path.split(txt_path)
        name= tail[:-4]  
        img_name=name+'.png'   
        img_path=os.path.join('./training/image_2/',img_name) # 路径需要自行修改              
        #print img_path  
        img_size=cv2.imread(img_path).shape
        data_height = img_size[0]
        data_width = img_size[1]    

        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

import transforms
from draw_box_utils import draw_objs
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random
import numpy as np

# read class_indict
category_index = {}
try:
    json_file = open('./pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

# load train data set
train_data_set = KITTIDataSet(os.getcwd(), data_transform["train"], "train.txt")
print(len(train_data_set))#自己修改的
for index in random.sample(range(0, len(train_data_set)), k=5):
    img, target = train_data_set[index]
    img = ts.ToPILImage()(img)
    draw_objs(img,
             target["boxes"].numpy(),
             target["labels"].numpy(),
             np.array([1 for i in range(len(target["labels"].numpy()))]),
             category_index = category_index,
             box_thresh=0.5,
             line_thickness=5)
    # plt.imshow(img)
    # plt.show()

    
