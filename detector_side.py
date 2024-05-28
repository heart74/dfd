import streamlit as st
from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Resize, Normalize

from backbone.model_selection import model_selection
from utils import convert_video
import time
#Loading the mtcnn model
detector = MTCNN(device='cuda:0').eval()
def usr_transforms(size=256):
        return Compose([
        Resize(height=size, width=size),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()]
    )

class VideoDataset(Dataset):
    def __init__(self, img_size, uploaded_file, istwo):
        self.img_size = img_size
        self.faces = []
        self.bboxes= []
        self.vids2imgs(uploaded_file)
        self.istwo=istwo
        self.length = len(self.faces)
    
    def __len__(self):
        return self.length
    
    def vids2imgs(self, uploaded_file):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        v_cap = cv2.VideoCapture(tfile.name)  # opencv打开文件
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.samples = np.linspace(0, v_len - 1, v_len//5).round().astype(int)
        i=0
        latest_iteration = st.empty()
        bar = st.progress(0)
        while (v_cap.isOpened()):
            success, vframe = v_cap.read()
            if success:
                if i in self.samples:
                    face,bbox = detect_faces(vframe)
                    self.faces.append(face)
                    self.bboxes.append(bbox)
            else:
                break
            i+=1
            bar.progress(int(i/v_len*100))
        v_cap.release()
        self.file = tfile.name
        self.frames = v_len

    def __getitem__(self, idx):
        # 定位到指定帧
        face = self.faces[idx]
        face_tensor = usr_transforms(self.img_size)(image=face)['image']
        if self.istwo:
            freq = face.copy()
            freq = cv2.cvtColor(freq, cv2.COLOR_RGB2YCR_CB)
            freq_tensor = usr_transforms(img_size)(image=freq)['image']
            face_tensor = torch.cat([face_tensor, freq_tensor], dim=0)
        return face_tensor

def test(model,loader,device,fea=False):
    model.eval()
    y_preds = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            pred = inference_single(model,data,device,fea)
            y_preds += list(pred)
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

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
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

# 人脸检测与裁剪、返回人像与Bounding Box
def detect_faces(image):
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    try:
        boxes, _ = detector.detect(image_pil)
        x, y, size = get_boundingbox(boxes.flatten(), width, height)
        cropped_face = image[y:y + size, x:x + size]
    except:
        st.error("未检测到人脸。")
        return (np.array([1,2,3]),0)
    return cropped_face,[x,y,size]

def check_type(upload):
    if upload.name.endswith('.jpg') or upload.name.endswith('.jpeg') or upload.name.endswith('.png'):
        return 'img'
    elif upload.name.endswith('.mp4'):
        return 'vid'
    else:
        raise ValueError("Invalid file type.")

#展示视频
def save_vids(tfilename,samples,preds,bboxs,save_path):
    v_cap = cv2.VideoCapture(tfilename)  # opencv打开文件
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    i,j=0,0
    frames = []
    while (v_cap.isOpened()):
        success, frame = v_cap.read()
        if success:
            pred = round(preds[j].item(),2)
            label = 'fake:'+str(pred) if pred>0.5 else "real:"+str(pred)
            color = (0,0,255) if pred>0.5 else (0,255,0)
            x1,x2,y1,y2=bboxs[j][0],bboxs[j][0]+bboxs[j][2],bboxs[j][1],bboxs[j][1]+bboxs[j][2]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
            cv2.putText(frame,label,(x1+90,y2+60),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
            if i in samples:
                j+=1
            frames.append(frame)
        else:
            break
        i+=1
    v_cap.release()
    # 确定新视频的参数
    output_path = save_path
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码格式
    # 使用与原视频相同的帧率创建VideoWriter对象
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # 将帧写入新视频
    for frame in frames:
        out.write(frame)
    out.release()  # 释放新视频文件
    convert_video(output_path,"./detect.mp4")

# ========================================================================== #
st.markdown("# <center> 深度伪造检测系统 :mag: </center>",True)
st.sidebar.write("## 加载模型 :floppy_disk:")
model_list = ['Xception', 'EfficientNet-B4', 'CDC']
model_name = st.sidebar.selectbox("选择检测模型", model_list)
model_name = "CDC"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model_name == 'CDC':
    model, input_size = model_selection('EFF_CMIA', freq_srm=(3,1),fusion="se",eff='efficientnet_b4',embeddings=0)
    img_size = max(input_size)
    checkpoint = './model_zoo/efficientnet_b4_Iter_037500_ACC94.57_AUC98.42.ckpt'
    state_dict = torch.load(checkpoint, map_location="cpu")['net_state_dict']
    model.load_state_dict(state_dict,strict=False)
    model = model.to(device)
    model.eval()
    st.sidebar.markdown("#### 模型加载成功 ",True)
    istwo = True
# else:
#     st.error(f"Model {model_name} not implemented.")

st.sidebar.write("## 上传待检测图像/视频 :movie_camera:")
uploaded_file = st.sidebar.file_uploader("选择图像或视频", type=["jpg", "png", "jpeg", "mp4"])
cnt1,cnt2=st.columns(2)
if uploaded_file:
    type = check_type(uploaded_file)
    with cnt1:
        if type=='vid':
            st.markdown("### <center> 上传视频 </center>",True)
            st.video(uploaded_file)
            st.write('检测进度')
            start_time = time.time()
            vid_dataset = VideoDataset(img_size, uploaded_file,istwo)
            st.write('视频帧数:',vid_dataset.frames)
            dataLoader = DataLoader(vid_dataset, batch_size=4, shuffle=False)
            samples = vid_dataset.samples
            bboxs = vid_dataset.bboxes
            tfilename = vid_dataset.file
            # print(tfilename)
            preds = test(model,dataLoader,device,fea=True)
            end_time = time.time()
            save_path = './temp_vids.mp4'
            save_vids(tfilename,samples,preds,bboxs,save_path)
            with cnt2:
                st.markdown("### <center> 检测结果 </center>",True)
                st.video("./detect.mp4")
                prob = round(sum(preds)/len(preds),2)
                if prob<0.5:
                    st.success("视频中的人脸为真实人脸，真实概率为"+str(1-prob),icon="✅")
                else:
                    st.error("视频中的人脸为伪造人脸，伪造概率为"+str(prob),icon="🚨")
            st.write('检测耗时:',round(end_time-start_time,2),'秒')
        else:
            st.markdown("### <center> 上传图片 </center>",True)
            st.image(uploaded_file)
            image_pil=Image.open(uploaded_file).convert('RGB')
            image_pil.save('tmp.png')
            image=cv2.imread('tmp.png')
            face, cord= detect_faces(image)
            if cord == 0:
                st.error("未检测到人脸。")
            else:
                rgb_tensor = usr_transforms(img_size)(image=face)['image']
                freq = face.copy()
                freq = cv2.cvtColor(freq, cv2.COLOR_RGB2YCR_CB)
                freq_tensor = usr_transforms(img_size)(image=freq)['image']
                input_tensor = torch.cat([rgb_tensor, freq_tensor], dim=0).unsqueeze(0)
                pred = inference_single(model,input_tensor,device,fea=True)
                pred = round(pred.item(),2)
                if pred>0.5:
                    label = 'fake:'+str(pred)
                    color = (0,0,255)
                else:
                    label="real:"+str(1-pred)
                    color = (0,255,0)
                x1,x2,y1,y2=cord[0],cord[0]+cord[2],cord[1],cord[1]+cord[2]
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                cv2.putText(image,label,(x1+80,y2+70),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
                with cnt2:
                    st.markdown("### <center> 检测结果 </center>",True)
                    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    st.image(image)
                    if pred<0.5:
                        st.success("图片中的人脸为真实人脸，真实概率为"+str(1-pred),icon="✅")
                    else:
                        st.error("图片中的人脸为伪造人脸，伪造概率为"+str(pred),icon="🚨")