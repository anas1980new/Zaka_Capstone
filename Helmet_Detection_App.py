

from requests import delete, get
import requests
import streamlit as st
import numpy as np
import cv2
import tempfile
import time
from PIL import Image
import glob

from streamlit_webrtc import webrtc_streamer




DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo.mp4'

st.title('Helmet Detection App')


#Model input -----------------------------------------------------------------------------------classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]
classes = ['head', 'helmet']


URL = "https://github.com/anas1980new/Zaka_Capstone/releases/download/lfs/best.pt" # Downloading Model weights
response = requests.get(URL)
open("best.pt", "wb").write(response.content)

opt  = {
    "weights": "best.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/custom_data.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.7, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : None  # list of classes to filter or None

}

# Seting up the Model envirment

import os
import sys
sys.path.append('/content/drive/MyDrive/TheCodingBug/yolov7')


import argparse
import time
from pathlib import Path
import cv2
import torch
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



#Detection Function Code ------------------------------------------------------------------------------------------------------------------
def detectit(frame):
  import glob, random

  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
      model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    img0 = frame
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment= False)[0]

    # Apply NMS
    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:

        classes.append(opt['classes'].index(class_name))


    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
    t2 = time_synchronized()
    for i, det in enumerate(pred):
      s = ''
      s += '%gx%g ' % img.shape[2:]  # print string
      gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
      if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for c in det[:, -1].unique():
          n = (det[:, -1] == c).sum()  # detections per class
          s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
      
        for *xyxy, conf, cls in reversed(det):

          label = f'{names[int(cls)]} {conf:.2f}'
          plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
  
  return(frame)










#Streamlit Application Part ---------------------------------------------------------------------------------------------------------



st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        wedith: 350px
    }

    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        wedith: 350px
        margin-lift: -350px
    }
    </style>

    """,
    unsafe_allow_html=True
)

st.sidebar.title('Helmet Detection Sidebar')
st.sidebar.subheader('paramaters')



app_mode = st.sidebar.selectbox('Test Me ! 👀 :) ',['Detect an Image', 'Detect a video', 'Detect on webcam','About App'])


#About App---------------------------------------------------------------------
if app_mode =="About App":
    st.markdown('In this App we are using the new **Yolov7** alghorithem to detect Safety helmets.')
    
    st.markdown('Google Colab Pro 😎 was used to train the model. 😤 was going to train for longer but no compute units left 😤')

    st.markdown('App made by:  Anas Salama')

    with st.expander('Click here to know more', expanded=False):
        st.write("""
        The chart above shows some numbers I picked for you. \n

        I rolled actual dice for these, so they're *guaranteed* to
        be random.
        """ )
        #st.image()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            wedith: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            wedith: 350px
            margin-lift: -350px
        }
        </style>

        """,
        unsafe_allow_html=True
    )

    #st.video('.......video Link.......')

#Detect an Image---------------------------------------------------------------------

elif app_mode == 'Detect an Image':
    st.sidebar.markdown('-----------')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][arial-expanded="true"] > div:first-child{
            wedith: 350px
        }

        [data-testid="stSidebar"][arial-expanded="false"] > div:first-child{
            wedith: 350px
            margin-lift: -350px
        }
        </style>

        """,
        unsafe_allow_html=True
    )

    st.markdown('**Detection**')
    
    image = None 

    file_buffer = st.sidebar.file_uploader("Upload an Image", type=["JPG", "jpeg", "png"])
    if file_buffer is not None:
        
        #resized = cv2.resize(file_buffer, (640,640), interpolation = cv2.INTER_NEAREST)
        image = np.array(Image.open(file_buffer))
        feedback1="Uploaded Image: "
        st.sidebar.image(image)

    else:
        st.text('You need to uplad an image or use a predifined demo image from below')
        Use_demo = st.button('Use Demo image')
        
        if Use_demo:
            all_img_in_file = glob.glob("data/test/images/*")
            rnd_img = random.choice(all_img_in_file)
            #source_image_path = random.choice(rnd_img)

            demo_image = DEMO_IMAGE
            image = np.array(Image.open(rnd_img))
            feedback1= "No Image Uploaded!\nUsing Demo Image..."

    if image is not None:
        detectedframe = detectit(image)
        st.image(detectedframe)


  

#Detect on Webcam---------------------------------------------------------------------

# elif app_mode == 'Detect on webcam':
#     st.set_option('deprecation.showfileUploaderEncoding', False)

#     use_webcam = st.sidebar.button(' Use Webcam')
#     record = st.sidebar.checkbox("Record Video")

#     if record:
#         st.checkbox('Recording in progress 🧑‍🦱', value=True)

#     st.markdown(
#         """
#         <style>
#         [data-testid="stSidebar"][arial-expanded="true"] > div:first-child{
#             wedith: 350px
#         }

#         [data-testid="stSidebar"][arial-expanded="false"] > div:first-child{
#             wedith: 350px
#             margin-lift: -350px
#         }
#         </style>

#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown('**Detection**')
    
#     #stframe = st.empty
#     video_file_buffer = st.sidebar.file_uploader("Upload a video", type = ["mp3", "mov", "avi", "asf", "m4v"])
#     tffile = tempfile.NamedTemporaryFile(delete=False)

#     ##Video Input
#     if not video_file_buffer:
#         if use_webcam:
#             vid = webrtc_streamer(key="sample")
#         else:
#             vid = cv2.VideoCapture(DEMO_VIDEO)
#             tffile.name = DEMO_VIDEO

#     else:
#         tffile.write(video_file_buffer.read())
#         vid = cv2.VideoCapture(tffile.name)

#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps_input = int(vid.get(cv2.CAP_PROP_FPS))

#     #Recording Code
#     codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

#     st.sidebar.text('Input Video')
#     st.sidebar.video(tffile.name)


#     class VideoProcessor:
#         def __init__(self) -> None:
#             pass

#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")


# ctx = webrtc_streamer(
#     key="example",
#     video_processor_factory=VideoProcessor,
#     rtc_configuration={
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }
# )
# if ctx.video_processor:
#     ctx.video_processor.threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
#     ctx.video_processor.threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)


#     fps = 0
#     i = 0 

#     # while True:
#     # # Get frames
#     #     ret, frame= vid.read()

#     #     frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        
#     #     cv2.imshow("Frame", frame)
#     #     # the 'q' button is set as the
#     #     # quitting button you may use any
#     #     # desired button of your choice
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         cv2.destroyWindow('Frame')
#     #         break


#     while True:
#         _, frame = vid.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         #FRAME_WINDOW.image(frame)
#     else:
#         st.write('Stopped')



#     webrtc_streamer(key="sample")








    # #Intialize Camera

    # cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # # Create Window
    # cv2.namedWindow("Frame")
    # #cv2.setMouseCallback("Frame", click_button)



    # while True:
    #     # Get frames
    #     ret, frame= cap.read()

    #     # Get active button list
    #     #active_buttons = button.active_buttons_list()
    #     #print(active_buttons)


    #    #

    #     cv2.imshow("Frame", frame)
    #     # the 'q' button is set as the
    #     # quitting button you may use any
    #     # desired button of your choice
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cv2.destroyWindow('Frame')
    #         break

   
