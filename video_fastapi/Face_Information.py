import os
import cv2
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from preprocessing import faceAlignment
from PIL import Image
import torch.nn.functional as F
import math

from deep_model import *

from moviepy.editor import *
import librosa

# device 설정. CUDA 사용가능하면 CUDA 모드로, 못 쓰면 CPU 모드로 동작
# 단 cpu로 연산 할 경우 인식 함수 내 코드 수정 필요.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cuda"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 얼굴 정보를 담을 class
class Face:
    def __init__(self):
        # class init
        self.rt = [-1, -1, -1, -1]
        self.sID = ""
        self.fvScore = -1.
        self.ptLE = [-1, -1]
        self.ptRE = [-1, -1]
        self.ptLM = [-1, -1]
        self.ptRM = [-1, -1]

        self.ptLED = [-1, -1]
        self.ptRED = [-1, -1]

        self.fYaw = -1.
        self.fPitch = -1.
        self.fRoll = -1.

        self.nEmotion = -1
        self.fEmotionScore = [-1, -1, -1, -1, -1, -1, -1]



def get_state_dict(origin_dict):
    old_keys = origin_dict.keys()
    new_dict = {}

    for ii in old_keys:
        temp_key = str(ii)
        if temp_key[0:7] == "module.":
            new_key = temp_key[7:]
        else:
            new_key = temp_key

        new_dict[new_key] = origin_dict[temp_key]
    return new_dict



# 초기화
def Initialization():
    # face detector. OpenCV SSD
    FD_Net = cv2.dnn.readNetFromCaffe("C:/Users/withmind/Desktop/models/opencv_ssd.prototxt", "C:/Users/withmind/Desktop/models/opencv_ssd.caffemodel")


    # Landmark 모델
    Landmark_Net = LandmarkNet(3, 3)
    # # Landmark_Net = torch.nn.DataParallel(Landmark_Net).to(device)
    Landmark_Net = Landmark_Net.to(device)
    Landmark_Net.load_state_dict(torch.load("C:/Users/withmind/Desktop//models/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])
    # Landmark_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])


    # Headpose 모델
    Headpose_Net = HeadposeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    Headpose_Net = Headpose_Net.to(device)
    Headpose_Net.load_state_dict(torch.load("C:/Users/withmind/Desktop/models/ETRI_HEAD_POSE.pth.tar"))
    # Headpose_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_HEAD_POSE.pth.tar"))

    # Emotion classifier
    Emotion_Net = EmotionNet(num_classes=7).to(device)
    new_dict = get_state_dict(torch.load("C:/Users/withmind/Desktop/models/ETRI_Emotion.pth.tar")['state_dict'])
    # # new_dict = get_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_EMOTION.pth.tar")['state_dict'])
    Emotion_Net.load_state_dict(new_dict)



    # 각 모델 evaluation 모드로 설정
    Landmark_Net.eval()
    Headpose_Net.eval()
    Emotion_Net.eval()

    return FD_Net, Landmark_Net, Headpose_Net, Emotion_Net


# 얼굴검출
# OpenCV 기본 예제 적용
def Face_Detection(FD_Net, cvImg, list_Face):
    del list_Face[:]
    img = cvImg.copy()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    FD_Net.setInput(blob)
    detections = FD_Net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.95:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ETRIFace 클래스에 입력하여 리스트에 저장.
            # 정방 사이즈로 조절하여 저장 함.
            ef = Face()
            difX = endX - startX
            difY = endY - startY
            if difX > difY:
                offset = int((difX - difY) / 2)
                new_startY = max(startY - offset, 0)
                new_endY = min(endY + offset, h - 1)
                new_startX = max(startX, 0)
                new_endX = min(endX, w - 1)
                ef.rt = [new_startX, new_startY, new_endX, new_endY]
            else:
                offset = int((difY - difX) / 2)
                new_startX = max(startX - offset, 0)
                new_endX = min(endX + offset, w - 1)
                new_startY = max(startY, 0)
                new_endY = min(endY, h - 1)
                ef.rt = [new_startX, new_startY, new_endX, new_endY]

            list_Face.append(ef)

    # torch.cuda.empty_cache()

    return len(list_Face)

# landmark 검출
def Landmark_Detection(Landmark_Net, cvImg, list_Face, nIndex):
    h, w, _ = cvImg.shape
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yLeftBottom_in = list_Face[nIndex].rt[1]
    yRightTop_in = list_Face[nIndex].rt[3]
    xLeftBottom_in = list_Face[nIndex].rt[0]
    xRightTop_in = list_Face[nIndex].rt[2]

    n15 = (yRightTop_in - yLeftBottom_in) * 0.2
    xLeftBottom_in = max(xLeftBottom_in - n15, 0)
    xRightTop_in = min(xRightTop_in + n15, w-1)
    yLeftBottom_in = max(yLeftBottom_in - n15, 0)
    yRightTop_in = min(yRightTop_in + n15, h-1)
    INPUT = cvImg[(int(yLeftBottom_in)):(int(yRightTop_in)), (int(xLeftBottom_in)): (int(xRightTop_in))]

    # 인식 좌표 정보에 얼굴 위치 보정하기 위한 값
    # offsetX = list_ETRIFace[nIndex].rt[0]
    # offsetY = list_ETRIFace[nIndex].rt[1]
    offsetX = xLeftBottom_in
    offsetY = yLeftBottom_in

    # preprocessing
    w = xRightTop_in - xLeftBottom_in
    INPUT = cv2.resize(INPUT, (256, 256))
    INPUT = INPUT / 255
    ratio = w / 256
    INPUT = np.transpose(INPUT, axes=[2, 0, 1])
    INPUT = np.array(INPUT, dtype=np.float32)
    INPUT = torch.from_numpy(INPUT)
    INPUT = torch.unsqueeze(INPUT, 0)
    INPUT = INPUT.to(device)
    OUTPUT = Landmark_Net(INPUT)
    OUTPUT = torch.squeeze(OUTPUT)
    output_np = OUTPUT.cpu().detach().numpy()
    output_np = output_np * 1.1 * 256
    output_np = output_np * ratio

    # 좌표 보정
    for ii in range(68):
        output_np[ii * 2 + 0] = output_np[ii * 2 + 0] + offsetX
        output_np[ii * 2 + 1] = output_np[ii * 2 + 1] + offsetY

    leX = leY = reX = reY = lmX = lmY = rmX = rmY = nX = nY = 0
    for ii in range(36, 42, 1):
        leX = leX + output_np[ii * 2 + 0]
        leY = leY + output_np[ii * 2 + 1]

    for ii in range(42, 48, 1):
        reX = reX + output_np[ii * 2 + 0]
        reY = reY + output_np[ii * 2 + 1]

    # 눈, 입 양 끝점 저장
    list_Face[nIndex].ptLE = [int(leX / 6), int(leY / 6)]
    list_Face[nIndex].ptRE = [int(reX / 6), int(reY / 6)]
    list_Face[nIndex].ptLM = [int(output_np[48 * 2 + 0]), int(output_np[48 * 2 + 1])]
    list_Face[nIndex].ptRM = [int(output_np[54 * 2 + 0]), int(output_np[54 * 2 + 1])]
    list_Face[nIndex].ptN = [int(output_np[30 * 2 + 0]), int(output_np[30 * 2 + 1])]

    torch.cuda.empty_cache()

    return output_np


transformations_emotionnet = transforms.Compose(
    [transforms.Grayscale(),
     transforms.CenterCrop(128),
     transforms.ToTensor()]
)

transformations_headposenet = transforms.Compose(
    [transforms.Scale(224),
     transforms.CenterCrop(224), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def HeadPose_Estimation(HeadPose_Net, cvImg, list_Face, nIndex):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oImg = cvImg[list_Face[nIndex].rt[1]:list_Face[nIndex].rt[3], \
          list_Face[nIndex].rt[0]:list_Face[nIndex].rt[2]].copy()

    # cv2.imshow("ZZ", oImg)
    # cv2.waitKey(0)

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    PILImg = Image.fromarray(oImg)

    img = transformations_headposenet(PILImg)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).to(device)

    yaw, pitch, roll = HeadPose_Net(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)

    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    list_Face[nIndex].fYaw = yaw_predicted
    list_Face[nIndex].fPitch = pitch_predicted
    list_Face[nIndex].fRoll = roll_predicted

    torch.cuda.empty_cache()

    return (yaw_predicted, pitch_predicted, roll_predicted)



def list2SoftList(srcList):
    tmpList = srcList.copy()

    fSum = 0.

    for ii in range(len(srcList)):
        fExp = np.exp(srcList[ii])
        fSum = fSum + fExp
        tmpList[ii] = fExp
    for ii in range(len(srcList)):
        srcList[ii] = tmpList[ii] / fSum;

    return srcList

def Emotion_Classification(Emotion_Net, cvImg, list_Face, nIndex):

    if list_Face[nIndex].ptLE == [-1, -1]:
        return -1

    img = cvImg.copy()

    ROIImg = faceAlignment(img, list_Face[nIndex].ptLE, list_Face[nIndex].ptRE
                           , list_Face[nIndex].ptLM, list_Face[nIndex].ptLM)

    PILROI = Image.fromarray(ROIImg)

    transformedImg = transformations_emotionnet(PILROI)
    transformedImg = torch.unsqueeze(transformedImg, 0)
    transformedImg = transformedImg.to(device)
    output_glasses = Emotion_Net(transformedImg)

    output_cpu = output_glasses.cpu().detach().numpy().squeeze()
    output = list2SoftList(output_cpu)
    output = output.tolist()

    # emotion label
    # surprise, fear, disgust, happy, sadness, angry, neutral
    list_Face[nIndex].nEmotion = output.index(max(output))
    for ii in range(7):
        list_Face[nIndex].fEmotionScore[ii] = output[ii]

    torch.cuda.empty_cache()

def Gaze_Regression(list_Face, nIndex):
    if list_Face[nIndex].ptLE == [-1, -1] or list_Face[nIndex].fYaw == -1:
        return -1

    d2r = 3.141592 / 180.0
    #fDist = math.sqrt(pow(list_Face[nIndex].ptRE[0] - list_Face[nIndex].ptLE[0], 2) + pow(list_Face[nIndex].ptRE[1] - list_Face[nIndex].ptLE[1], 2))
    fDist = 20 * ((list_Face[nIndex].fPitch + list_Face[nIndex].fYaw)/2)

    normX = -1 * math.sin(d2r * list_Face[nIndex].fYaw) * fDist
    normY = -1 * math.sin(d2r * list_Face[nIndex].fPitch) * math.cos(d2r * list_Face[nIndex].fYaw) * fDist

    list_Face[nIndex].ptLED = [list_Face[nIndex].ptLE[0] + normX, list_Face[nIndex].ptLE[1] + normY]
    list_Face[nIndex].ptRED = [list_Face[nIndex].ptRE[0] + normX, list_Face[nIndex].ptRE[1] + normY]

    torch.cuda.empty_cache()

    return list_Face[nIndex].ptLED, list_Face[nIndex].ptRED


import mediapipe as mp

class hand_Detector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.85, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_Hand_Position(self, img, handNo=0, draw=True):

        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmlist

    def fingersUp(self):
        fingers = []

        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)


        for id in range(0, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

class pose_Detector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.7, trackCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        # print(results.pose_landmarks)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def soundcheck(self):

    sound = AudioFileClip(self)  # self = .mp4

    shortsound = sound.subclip("00:00:01", "00:00:15")  # audio from 1 to 15 seconds
    fileroute = 'C:/Users/withmind/Desktop/models/'
    filename = 'sound.wav'
    shortsound.write_audiofile(fileroute + filename, 44100, 2, 2000, "pcm_s32le")

    y, sr = librosa.load(fileroute + filename)
    sound_result = 0
    for i in y:
        if y[-0] == 0.00:
            print('음성확인 > ', False)
            sound_result = False
            break
        else:
            if i == 0.00:
                continue
            else:
                print('음성확인 > ', True)
                sound_result = True
                break

    os.remove(fileroute + filename)

    return sound_result