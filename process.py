import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize
from PIL import Image
import cv2

from backbone.model_selection import model_selection

from facenet_pytorch import MTCNN

from tqdm import tqdm
import os
from io import BytesIO
import tempfile
import streamlit as st

#image_placeholder.image(to_show, caption='Video')  # 将图片帧展示在同一位置得到视频效果

# 处理上传的数据，jpg、png或mp4
def load_data(uploaded_file, model_name='cdc'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载MTCNN模型
    mtcnn = MTCNN(device='cuda:0').eval()
    if model_name == 'cdc':
        model, input_size = model_selection('EFF_CMIA', freq_srm=(3,1),fusion="se",eff='efficientnet_b4',embeddings=0)
        checkpoint = '/media/sdd/zhy/paper/model/te4_fs31_se_cmia_auc_113_ffpp_c23_aug_0302_095328/efficientnet_b4_Iter_037500_ACC94.57_AUC98.42.ckpt'
        state_dict = torch.load(checkpoint, map_location="cpu")['net_state_dict']
        model.load_state_dict(state_dict,strict=False)
        model = model.to(device)
        model.eval()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    st.write('model loaded')
    img_size = max(input_size)
    usr_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.45], std=[0.5, 0.5, 0.5]),
    ])
    try:
        # 检查文件类型并处理
        if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.png'):
            # 处理图像文件
            # 这里添加人脸检测与裁剪的代码
            croped_image,bbox = detect_faces(uploaded_file,mtcnn)
            # 将裁剪出来的人脸图像转换为Pytorch Tensor
            st.write('get croped image')
            face_tensor = usr_transform(croped_image)
            # 进行推理
            st.write('start inference')
            pred = inference_single(model,face_tensor,device)
            return pred, bbox
        elif uploaded_file.name.endswith('.mp4'):
            # 处理视频文件
            # 构建视频数据集
            vids_dataset = VideoDataset(max(input_size), uploaded_file, mtcnn)
            dataloader = DataLoader(vids_dataset, batch_size=1, shuffle=False)
            croped_images = vids_dataset.faces
            bboxs = vids_dataset.bboxes
            preds = test(model,dataloader,device)
            return preds, bboxs
        else:
            st.error("Unsupported file format. Please upload jpg, png, or mp4 files.")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
    
    
def test(model,loader,device,fea=False):
    model.eval()
    y_preds = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            pred = inference_single(model,data,device,fea)
            y_preds += pred
    return y_preds

def inference_single(model,tensor,device,fea=False):
    with torch.no_grad():
        inputs = tensor.to(device)
        if fea:
            _,logit = model(inputs)
        else:
            logit = model(inputs)
        prob = torch.softmax(logit, dim=1)
        y_pred = prob[:, 1].cpu().numpy()
    return y_pred

# 人脸检测与裁剪、返回人像与Bounding Box
def detect_faces(uploaded_file,mtcnn):
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_arr = np.asarray(image_pil)
    print(image_arr.shape)
    # image_arr = load_local_image(uploaded_file)
    # st.write(type(image_arr))
    # st.write(image_arr.shape)
    height, width = image_arr.shape[:2]
    st.write('start detect')
    image_pil = Image.fromarray(image_arr)
    st.write(type(image_pil))
    # st.write(image_pil.shape)
    try:
        boxes, _ = mtcnn.detect(image_pil)
        x, y, size = get_boundingbox(boxes.flatten(), width, height)
        cropped_face = image_arr[y:y + size, x:x + size]
    except:
        print("No face detected")
        st.write("No face detected")
        return None,None
    return cropped_face,[x,y,size]

# 从视频文件中提取帧
def vid2imgs(uploaded_file,mtcnn):
    # image_placeholder = st.empty()  # 创建空白块使得图片展示在同一位置
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    v_cap = cv2.VideoCapture(tfile.name)  # opencv打开文件
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (v_cap.isOpened() == False):
        st.write("读取视频文件时出现错误")
    faces = []
    bboxes= []
    while (v_cap.isOpened()):
        success, vframe = v_cap.read()
        if success:
            face,bbox = detect_faces(vframe,mtcnn)
            faces.append(face)
            bboxes.append(bbox)
        else:
            break
    v_cap.release()
    return faces,bboxes

class VideoDataset(Dataset):
    def __init__(self, img_size, uploaded_file, mtcnn):
        self.img_size = img_size
        self.mtcnn = mtcnn
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.faces = []
        self.bboxes= []
        self.vids2imgs(uploaded_file,mtcnn)
    
    def __len__(self):
        return self.length
    
    def vids2imgs(self, uploaded_file, mtcnn):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        v_cap = cv2.VideoCapture(tfile.name)  # opencv打开文件
        while (v_cap.isOpened()):
            success, vframe = v_cap.read()
            if success:
                face,bbox = detect_faces(vframe,mtcnn)
                self.faces.append(face)
                self.bboxes.append(bbox)
            else:
                break
        v_cap.release()
    
    def transforms(self, size=256):
        return Compose([
        Resize(height=size, width=size),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()]
        )

    def __getitem__(self, idx):
        # 定位到指定帧
        face = self.faces[idx]
        face_tensor = self.transforms(self.img_size)(image=face)['image']
        return face_tensor

# 图像预处理
# def transforms(size=256):
#     return Compose([
#         Resize(height=size, width=size),
#         Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#         ToTensorV2()
#     ]
#     )
    
@st.cache_data(show_spinner=False)
def load_local_image(uploaded_file):
    bytes_data = uploaded_file.getvalue()  
    image = np.array(Image.open(BytesIO(bytes_data)))
    return image

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb