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

#병렬처리
import multiprocessing as MP

# 사용가능한 CPU 개수 확인
print('가능한_CPU >> ', MP.cpu_count())



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

import mediapipe as mp

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

    def gaze_Detecor(self, img_show):
        if self != None:
            # print("gaze", gaze)
            center_gaze_x = (self[0][0] + self[1][0]) / 2
            center_gaze_y = (self[0][1] + self[1][1]) / 2
            cv2.circle(img_show, (int(center_gaze_x), int(center_gaze_y)), 8, (0, 0, 255), -1)
            center_gaze = (int(center_gaze_x), int(center_gaze_y))
            return center_gaze

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList


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
class ImConfig():
    name = 'im_video'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    FD_Net = cv2.dnn.readNetFromCaffe("/Users/tamy/Desktop/Tamy/coding/models/opencv_ssd.prototxt", "/Users/tamy/Desktop/Tamy/coding/models/opencv_ssd.caffemodel")

    Landmark_Net = LandmarkNet(3, 3)
    # Landmark_Net = torch.nn.DataParallel(Landmark_Net).to(device)
    Landmark_Net = Landmark_Net.to(device)
    Landmark_Net.load_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])

    Headpose_Net = HeadposeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    Headpose_Net = Headpose_Net.to(device)
    Headpose_Net.load_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_HEAD_POSE.pth.tar"))

    Emotion_Net = EmotionNet(num_classes=7).to(device)
    new_dict = get_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_EMOTION.pth.tar")['state_dict'])
    # new_dict = get_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_EMOTION.pth.tar")['state_dict'])
    Emotion_Net.load_state_dict(new_dict)

def Initialization():
    # face detector. OpenCV SSD
    # FD_Net = cv2.dnn.readNetFromCaffe("/Users/tamy/Desktop/Tamy/coding/models/opencv_ssd.prototxt", "/Users/tamy/Desktop/Tamy/coding/models/opencv_ssd.caffemodel")
    # FD_Net = cv2.dnn.readNetFromCaffe("/home/ubuntu/projects/withmind_video/im_video/file/opencv_ssd.prototxt", "/home/ubuntu/projects/withmind_video/im_video/file/opencv_ssd.caffemodel")
    FD_Net = ImConfig.FD_Net

    # Landmark 모델
    # Landmark_Net = LandmarkNet(3, 3)
    # # Landmark_Net = torch.nn.DataParallel(Landmark_Net).to(device)
    # Landmark_Net = Landmark_Net.to(device)
    # Landmark_Net.load_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])
    # Landmark_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_LANDMARK_68pt.pth.tar", map_location=device)['state_dict'])
    ImConfig.Landmark_Net

    # Headpose 모델
    # Headpose_Net = HeadposeNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # Headpose_Net = Headpose_Net.to(device)
    # Headpose_Net.load_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_HEAD_POSE.pth.tar"))
    # Headpose_Net.load_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_HEAD_POSE.pth.tar"))
    ImConfig.Headpose_Net

    # Emotion classifier
    # Emotion_Net = EmotionNet(num_classes=7).to(device)
    # new_dict = get_state_dict(torch.load("/Users/tamy/Desktop/Tamy/coding/models/ETRI_Emotion.pth.tar")['state_dict'])
    # # new_dict = get_state_dict(torch.load("/home/ubuntu/projects/withmind_video/im_video/file/ETRI_EMOTION.pth.tar")['state_dict'])
    # Emotion_Net.load_state_dict(new_dict)
    ImConfig.Emotion_Net

    # 각 모델 evaluation 모드로 설정
    # Landmark_Net.eval()
    # Headpose_Net.eval()
    # Emotion_Net.eval()

    ImConfig.Landmark_Net.eval()
    ImConfig.Headpose_Net.eval()
    ImConfig.Emotion_Net.eval()

    return FD_Net, ImConfig.Landmark_Net, ImConfig.Headpose_Net, ImConfig.Emotion_Net


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

# 감정 분석
def Emotion_analysis(Emotion_list):
    Emotion_surprise = 0
    Emotion_fear = 0
    Emotion_disgust = 0
    Emotion_happy = 0
    Emotion_sadness = 0
    Emotion_angry = 0
    Emotion_neutral = 0

    for i in range(len(Emotion_list)):
        Emotion_surprise = Emotion_surprise + Emotion_list[i][0]
        Emotion_fear = Emotion_fear + Emotion_list[i][1]
        Emotion_disgust = Emotion_disgust + Emotion_list[i][2]
        Emotion_happy = Emotion_happy + Emotion_list[i][3]
        Emotion_sadness = Emotion_sadness + Emotion_list[i][4]
        Emotion_angry = Emotion_angry + Emotion_list[i][5]
        Emotion_neutral = Emotion_neutral + Emotion_list[i][6]

        if Emotion_list != 0:
            Emotion_surprise_mean = Emotion_surprise * 100 / len(Emotion_list)
            Emotion_fear_mean = Emotion_fear * 100 / len(Emotion_list)
            Emotion_disgust_mean = Emotion_disgust * 100 / len(Emotion_list)
            Emotion_happy_mean = Emotion_happy * 100 / len(Emotion_list)
            Emotion_sadness_mean = Emotion_sadness * 100 / len(Emotion_list)
            Emotion_angry_mean = Emotion_angry * 100 / len(Emotion_list)
            Emotion_neutral_mean = Emotion_neutral * 100 / len(Emotion_list)

    Emotion_analysi_sum = Emotion_surprise_mean + Emotion_fear_mean + Emotion_disgust_mean + Emotion_happy_mean \
                          + Emotion_sadness_mean + Emotion_angry_mean + Emotion_neutral_mean

    return Emotion_surprise_mean, Emotion_fear_mean, Emotion_disgust_mean, Emotion_happy_mean, \
           Emotion_sadness_mean, Emotion_angry_mean, Emotion_neutral_mean, Emotion_analysi_sum


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


def hand_Detecor(self):
    # 손 체크 리스트
    Hand_list = []
    Hand_count = 0
    Hand_time_list = []
    Hand_point_list = []
    Hand_point_result = []

    if len(self) != 0:
        Hand_list.append(1)
        Hand_time_list.append(1)
        Hand_point_list.append(self[12])
    else:
        if len(self) > 3:
            Hand_count += 1
            print("손 연속 >> ", Hand_count)
            Hand_point_result.append(Hand_point_list)

    Hand_list = []
    Hand_point_list = []

    if len(self) != 0:
        # print("손 >> ", self[4])

        return Hand_list, Hand_count, Hand_time_list, Hand_point_list, Hand_point_result

def shoulder_vertically_left(left_shoulder, Landmark_list):
    # 랜드마크 리스트(5번) / 어깨 위아래 움직임 체크 기준
    landmark_no5_y = Landmark_list[9]
    Left_shoulder_move_list = []
    Left_shoulder_move_count = 0


    if left_shoulder[1] >= landmark_no5_y:
        Left_shoulder_move_list.append(left_shoulder[1])
    else:
        if len(Left_shoulder_move_list) > 3:
            Left_shoulder_move_count += 1

    return Left_shoulder_move_count



#어깨 상하
def shoulder_vertically_right(right_shoulder, Landmark_list):
    # 랜드마크 리스트(5번) / 어깨 위아래 움직임 체크 기준
    landmark_no5_y = Landmark_list[9]

    Right_shoulder_move_list = []
    Right_shoulder_move_count = 0

    if right_shoulder[1] >= landmark_no5_y:
        Right_shoulder_move_list.append(right_shoulder[1])
    else:
        if len(Right_shoulder_move_list) > 3:
            Right_shoulder_move_count += 1

        return Right_shoulder_move_count




# 어깨 좌우 움직임 count
def shoulder_horizontality_count(center_shoulder_left, Landmark_list):
    # 랜드마크 리스트(1번, 17번) / 어깨 좌우 움직임 체크 기준
    landmark_no1_x = Landmark_list[0]
    landmark_no17_x = Landmark_list[32]

    Center_shoulder_left_move_list = []
    Center_shoulder_left_move_count = 0

    if center_shoulder_left <= landmark_no1_x:
        Center_shoulder_left_move_list.append(center_shoulder_left)
    else:
        if len(Center_shoulder_left_move_list) > 3:
            Center_shoulder_left_move_count += 1

    Center_shoulder_right_move_list = []
    Center_shoulder_right_move_count = 0

    if center_shoulder_left >= landmark_no17_x:
        Center_shoulder_right_move_list.append(center_shoulder_left)
    else:
        if len(Center_shoulder_right_move_list) > 3:
            Center_shoulder_right_move_count += 1

    return Center_shoulder_left_move_count, Center_shoulder_right_move_count


# 어깨 기울기
def shoulder_slope(right_shoulder, left_shoulder):
    if right_shoulder[0] != left_shoulder[0]:
        shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (right_shoulder[0] - left_shoulder[0])
    else:
        shoulder_slope = 0

    return shoulder_slope

# 얼굴 각도 결과
def Roll_mean(Roll_list):
    Roll_sum = 0
    for ii in range(len(Roll_list)):
        Roll_sum = Roll_sum + Roll_list[ii]
    Roll_mean = Roll_sum / len(Roll_list)

    return Roll_mean

# 어깨 각도 결과
def Shoulder_slope_mean(Shoulder_slope_list):
    Shoulder_slope_sum = 0
    for iii in range(len(Shoulder_slope_list)):
        Shoulder_slope_sum = Shoulder_slope_sum + Shoulder_slope_list[iii]
    Shoulder_slope_mean = Shoulder_slope_sum / len(Shoulder_slope_list)
    # print("어깨 각도 평균", Shoulder_slope_mean)

    return Shoulder_slope_mean

# 어깨 움직임 결과
# print("왼쪽어", (Left_shoulder_list))
def Left_shoulder_max_y(Left_shoulder_list):
    Left_shoulder_max_y = max(t[1] for t in Left_shoulder_list)
    for x, y in enumerate(Left_shoulder_list):
        if Left_shoulder_max_y in y:
            Left_shoulder_max = y
    # print(Left_shoulder_max)

    return Left_shoulder_max

def Left_shoulder_min_y(Left_shoulder_list):
    Left_shoulder_min_y = min(t[1] for t in Left_shoulder_list)
    for x, y in enumerate(Left_shoulder_list):
        if Left_shoulder_min_y in y:
            Left_shoulder_min = y
    # print(Left_shoulder_min)

    return Left_shoulder_min

# print("오른쪽어", Right_shoulder_list)
def Right_shoulder_max_y(Right_shoulder_list):
    Right_shoulder_max_y = max(t[1] for t in Right_shoulder_list)
    for x, y in enumerate(Right_shoulder_list):
        if Right_shoulder_max_y in y:
            Right_shoulder_max = y
    # print(Right_shoulder_max)

    return Right_shoulder_max

def Right_shoulder_min_y(Right_shoulder_list):
    Right_shoulder_min_y = min(t[1] for t in Right_shoulder_list)
    for x, y in enumerate(Right_shoulder_list):
        if Right_shoulder_min_y in y:
            Right_shoulder_min = y
    # print(Right_shoulder_min)

    return Right_shoulder_min

# print("가운데어", Center_shoulder_list)
def Center_shoulder_max_x(Center_shoulder_list):
    Center_shoulder_max_x = max(t[0] for t in Center_shoulder_list)
    for x, y in enumerate(Center_shoulder_list):
        if Center_shoulder_max_x in y:
            Center_shoulder_max = y
    # print(Center_shoulder_max)

    return Center_shoulder_max

def Center_shoulder_min_x(Center_shoulder_list):
    Center_shoulder_min_x = min(t[0] for t in Center_shoulder_list)
    for x, y in enumerate(Center_shoulder_list):
        if Center_shoulder_min_x in y:
            Center_shoulder_min = y
    # print(Center_shoulder_min)

    return Center_shoulder_min

# 제스처 시간
def Left_Hand_time_calculation(Left_hand_time_list):
    Left_Hand_time = float(len(Left_hand_time_list) / 6)

    return Left_Hand_time

def Right_Hand_time_calculation(Right_hand_time_list):
    Right_Hand_time = float(len(Right_hand_time_list) / 6)

    return Right_Hand_time




# 분석 기록2 CSV
import pandas as pd
import numpy as np
import csv

class Average:
    # Gaze 표준편차 / Gaze_Avg = [x_std, y_std] 리스트안 튜플
    def Gaze_Avg(self):
        df = pd.DataFrame(self, columns=['x', 'y'])
        x_std = round(np.std(df['x']))
        y_std = round(np.std(df['y']))

        GazeAvg_x = x_std
        GazeAvg_y = y_std

        return GazeAvg_x,  GazeAvg_y

    # 어깨 상하 최고 길이 도출
    def vertically_Avg(Left_shoulder_max, Left_shoulder_min, Right_shoulder_max, Right_shoulder_min):
        Left_shoulder = Left_shoulder_max[1] - Left_shoulder_min[1]
        Right_shoulder = Right_shoulder_max[1] - Right_shoulder_min[1]

        if Left_shoulder > Right_shoulder:
            return Left_shoulder
        else:
            return Right_shoulder

    # 어깨 좌우 최고 길이 도출
    def horizontally_Avg(Center_shoulder_max, Center_shoulder_min):
        horizontally = Center_shoulder_max[0] - Center_shoulder_min[0]

        return horizontally

    # 제스처 왼.오 큰값 도출
    def GestureTIME(Left_Hand_time, Right_Hand_time):
        Left_Hand = Left_Hand_time
        Right_Hand = Right_Hand_time

        if Left_Hand > Right_Hand:
            return Left_Hand
        else:
            return Right_Hand


    def Average_csv(Gaze_velue, Roll_velue, Shoulder_velue, shoulder_left_count, shoulder_right_count,vertically_value, horizontally_value, GestureTIME_value):
        # ftp = FTP()
        #
        # ftp.connect('withmind.cache.smilecdn.com', 21)
        # ftp.login('withmind', 'dnlemakdlsem1!')
        # ftp.cwd('./analy_result')
        filename = '/Average.csv'
        fileroute = '/Users/tamy/Desktop'

        # CSV 누적
        headersCSV = ['Gaze', 'Roll', 'Shoulder', 'shoulder_left_count', 'shoulder_right_count', 'vertically', 'horizontally', 'GestureTIME']
        dict = {'Gaze': Gaze_velue,
                'Roll': Roll_velue,
                'Shoulder': Shoulder_velue,
                'shoulder_left_count': shoulder_left_count,
                'shoulder_right_count': shoulder_right_count,
                'vertically': vertically_value,
                'horizontally': horizontally_value,
                'GestureTIME': GestureTIME_value
                }

        with open(fileroute + filename, mode='a', encoding='utf-8', newline='') as csvfile:
            wr = csv.DictWriter(csvfile, fieldnames=headersCSV)
            wr.writerow(dict)


# 시선 히트맵
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class heatmap:
    # self = Gaze_list -> Gaze_list_df
    def col2(Gaze_list):
        df = pd.DataFrame(Gaze_list, columns=['x', 'y'])
        return df

    def kde(Gaze_list_df):
        sns.kdeplot(
            Gaze_list_df['x'],
            Gaze_list_df['y'],
            shade=True,
            shade_lowest=False,
            alpha=.3,
            n_levels=30,
            cmap='Reds'
        )

    def kde_mean(Gaze_list_df):
        df = pd.DataFrame(Gaze_list_df, columns=['x', 'y'])

        df_mean_X, df_mean_Y = df['x'].mean(), df['y'].mean()
        int(df_mean_X)
        int(df_mean_Y)

        #df_mode_X, df_mode_Y = df['x'].mode(), df['y'].mode()

        sns.kdeplot(
            df['x'],
            df['y'],
            shade=True,
            shade_lowest=False,
            alpha=.7,
            n_levels=20,
            cmap='Reds')

        plt.scatter(df_mean_X, df_mean_Y, s=15, color="blue", alpha=0.5, label='mean point')
        #plt.scatter(df_mode_X, df_mode_Y, s=100, color="green", alpha=0.5, label='mean point')
        plt.title("heat map_ mean point", fontsize=15)
        plt.xlim([0, 640])
        plt.ylim([0, 480])

        plt.legend()
        '''
        plt.show(block=False)  # 5초후 삭제
        plt.pause(10)
        plt.close("all")
        '''
        plt.show()



def soundcheck(self):

    sound = AudioFileClip(self)  # self = .mp4

    shortsound = sound.subclip("00:00:01", "00:00:15")  # audio from 1 to 15 seconds
    fileroute = '/Users/tamy/Desktop/Tamy/coding/models/'
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

from ftplib import FTP

ftp = FTP()

def ftp_Analysis_json(userkey, videoNo):
    ftp.connect('withmind.cache.smilecdn.com', 21)
    ftp.login('withmind', 'dnlemakdlsem1!')
    ftp.cwd('./analy_result')
    filename = '%d_%d_result.json' % (int(userkey), int(videoNo))
    # myfile = open('C:/Users/withmind/Documents/withmind/IM/01_python_pytorch_project/01_python_pytorch_project/video/im_video/' + filename, 'wb')
    # with open('C:/Users/withmind/Documents/withmind/IM/01_python_pytorch_project/01_python_pytorch_project/video/im_video/' + filename) as contents:
    #     ftp.storbinary('STOR %s' % filename, contents)
    # fileroute = '/home/ubuntu/projects/withmind_video/im_video/'
    fileroute = '/Users/tamy/Desktop/Tamy/coding/'
    myfile = open(fileroute + filename, 'rb')
    ftp.storbinary('STOR ' + filename, myfile)

    myfile.close()
    os.remove(fileroute + filename)
    torch.cuda.empty_cache()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
