from fastapi import FastAPI
from pydantic import BaseModel
from Face_Information import *
import cv2
import copy
from starlette.responses import JSONResponse
import uvicorn
from ftplib import FTP
import json
from collections import OrderedDict

from fastapi import BackgroundTasks


from typing import List
from starlette.middleware.cors import CORSMiddleware
from database import session
from model import TestTable, Test
import asyncio





app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#
# FD_Net, Landmark_Net, Headpose_Net, Emotion_Net = Initialization()
# pose_detector = pose_Detector()
# Emotion_list = []



async def video_task(userkey, videoNo, videoaddress):
    FD_Net, Landmark_Net, Headpose_Net, Emotion_Net = Initialization()
    pose_detector = pose_Detector()
    vc = cv2.VideoCapture(videoaddress)
    FPS = cv2.CAP_PROP_FPS
    sound_confirm = soundcheck(videoaddress)

    Gaze_list = []
    Roll_list = []
    Emotion_list = []
    Left_Hand_count = 0
    Left_Hand_time_list = []
    Left_Hand_point_list = []
    Left_Hand_point_result = []
    Right_Hand_count = 0
    Right_Hand_time_list = []
    Right_Hand_point_list = []
    Right_Hand_point_result = []
    Left_shoulder_list = []
    Right_shoulder_list = []
    Center_shoulder_list = []
    Shoulder_slope_list = []
    shoulder_vertically_left_count = []
    shoulder_vertically_right_count = []

    bReady = True
    # print("1")
    if vc.isOpened() == False:
        bReady = False
        # print("errorrrrrrrrrr")

    while(bReady):
        # print("2")
        ret, frame = vc.read()
        if ret:
            if (int(vc.get(1)) % 5 == 0):
                # print("3")
                # print("frame", frame)
                frame = cv2.flip(frame, 1)
                img = frame
                img_show = copy.deepcopy(img)

                list_Face = []
                Face_count_list = []

                # face detection
                Face_Detection(FD_Net, img, list_Face)


                Face_count_list.append(len(list_Face))
                # print("체크체크", len(list_Face))
                if len(list_Face) > 0:
                    # print("4")
                    # draw face ROI. list_ETRIFace 를 순회하며 모든 검출된 얼굴의 박스 그리기
                    for ii in range(len(list_Face)):
                        cv2.rectangle(img_show, (list_Face[ii].rt[0], list_Face[ii].rt[1])
                                      , (list_Face[ii].rt[2], list_Face[ii].rt[3]), (0, 255, 0), 2)

                    # 이하 추가적인 인식은 첫번째 얼굴에 대해서만 수행.
                    # nIndex에 원하는 얼굴을 선택 가능

                    # landmark detection
                    Landmark_list = Landmark_Detection(Landmark_Net, img, list_Face, 0)
                    # print("landmark", landmark)

                    # draw landmark
                    # cv2.circle(img_show, (list_Face[0].ptLE[0], list_Face[0].ptLE[1]), 1, (0,0,255), -1)
                    # cv2.circle(img_show, (list_Face[0].ptRE[0], list_Face[0].ptRE[1]), 1, (0, 0, 255), -1)
                    # cv2.circle(img_show, (list_Face[0].ptLM[0], list_Face[0].ptLM[1]), 1, (0, 0, 255), -1)
                    # cv2.circle(img_show, (list_Face[0].ptRM[0], list_Face[0].ptRM[1]), 1, (0, 0, 255), -1)

                    # pose estimation
                    pose = HeadPose_Estimation(Headpose_Net, img, list_Face, 0)
                    # print("Y:%.1f / P:%.1f / R:%.1f" % (pose[0], pose[1], pose[2]))
                    Roll_list.append(pose[2].item())

                    # emotion classification
                    Emotion_Classification(Emotion_Net, img, list_Face, 0)
                    # emotion label
                    sEmotionLabel = ["surprise", "fear", "disgust", "happy", "sadness", "angry", "neutral"]
                    sEmotionResult = "Emotion : %s" % sEmotionLabel[list_Face[0].nEmotion]
                    EmotionResult = list_Face[0].fEmotionScore

                    Emotion_list.append(EmotionResult)

                    # 시선
                    gaze = Gaze_Regression(list_Face, 0)
                    Gaze_list.append(pose_Detector.gaze_Detecor(gaze, img_show))

                    # 동작 검출
                    pose_detector.findPose(img_show)
                    lmList_pose = pose_detector.findPosition(img_show)

                    # 왼손 추적 22222222
                    if lmList_pose != 0:
                        if lmList_pose[15][1] < 640 and lmList_pose[15][2] < 480:
                            Left_hand = [lmList_pose[15][1], lmList_pose[15][2]]
                            Left_Hand_point_list.append(Left_hand)
                        else:
                            if len(Left_Hand_point_list) > 3:
                                Left_Hand_count += 1
                                Left_Hand_point_result.extend(Left_Hand_point_list)
                            Left_Hand_point_list = []


                        # 오른손 추적 22222222
                        if lmList_pose[16][1] < 640 and lmList_pose[16][2] < 480:
                            Right_hand = (lmList_pose[16][1], lmList_pose[16][2])
                            Right_Hand_point_list.append(Right_hand)
                        else:
                            if len(Right_Hand_point_list) > 3:
                                Right_Hand_count += 1
                                Right_Hand_point_result.extend(Right_Hand_point_list)
                            Right_Hand_point_list = []

                        # 어깨
                        left_shoulder = (lmList_pose[11][1], lmList_pose[11][2])
                        right_shoulder = (lmList_pose[12][1], lmList_pose[12][2])
                        center_shoulder_left = int((lmList_pose[11][1] + lmList_pose[12][1]) / 2)
                        center_shoulder_right = int((lmList_pose[11][2] + lmList_pose[12][2]) / 2)
                        center_shoulder = (center_shoulder_left, center_shoulder_right)


                        Left_shoulder_list.append(left_shoulder)
                        Right_shoulder_list.append(right_shoulder)
                        Center_shoulder_list.append(center_shoulder)
                        # print('left_shoulder >> ', left_shoulder)
                        # print('landmark >> ', Landmark_list)

                        # 어깨 상하
                        shoulder_vertically_left_count.append(shoulder_Detector.shoulder_vertically_left(left_shoulder, Landmark_list))
                        shoulder_vertically_right_count.append(shoulder_Detector.shoulder_vertically_right(right_shoulder, Landmark_list))

                        # 어깨 좌우
                        shoulder_horizontality_count_value = shoulder_Detector.shoulder_horizontality_count(center_shoulder_left, Landmark_list)

                        # 어깨 기울기
                        shoulder_slope_value = shoulder_Detector.shoulder_slope(right_shoulder, left_shoulder)
                        Shoulder_slope_list.append(shoulder_slope_value)
        else:
            break

    Face_count_no_one = len(Face_count_list) - Face_count_list.count(1)
    # print(Face_count_no_one)
    if Face_count_no_one * 5 >= (FPS * 7):
        # print("분석 ㄴㄴ")
        Face_analy_result = 1
    else:
        # print("분석 ok")
        Face_analy_result = 0

    # 감정 분석 결과 연산

    Emotion_surprise = 0
    Emotion_fear = 0
    Emotion_disgust = 0
    Emotion_happy = 0
    Emotion_sadness = 0
    Emotion_angry = 0
    Emotion_neutral = 0

    # print(len(Emotion_list))
    for i in range(len(Emotion_list)):
        Emotion_surprise = Emotion_surprise + Emotion_list[i][0]
        Emotion_fear = Emotion_fear + Emotion_list[i][1]
        Emotion_disgust = Emotion_disgust + Emotion_list[i][2]
        Emotion_happy = Emotion_happy + Emotion_list[i][3]
        Emotion_sadness = Emotion_sadness + Emotion_list[i][4]
        Emotion_angry = Emotion_angry + Emotion_list[i][5]
        Emotion_neutral = Emotion_neutral + Emotion_list[i][6]
    Emotion_surprise_mean = Emotion_surprise * 100 / len(Emotion_list)
    Emotion_fear_mean = Emotion_fear * 100 / len(Emotion_list)
    Emotion_disgust_mean = Emotion_disgust * 100 / len(Emotion_list)
    Emotion_happy_mean = Emotion_happy * 100 / len(Emotion_list)
    Emotion_sadness_mean = Emotion_sadness * 100 / len(Emotion_list)
    Emotion_angry_mean = Emotion_angry * 100 / len(Emotion_list)
    Emotion_neutral_mean = Emotion_neutral * 100 / len(Emotion_list)
    # print("놀람", Emotion_surprise_mean, "%")
    # print("공포", Emotion_fear_mean, "%")
    # print("역겨움", Emotion_disgust_mean, "%")
    # print("행복", Emotion_happy_mean, "%")
    # print("슬픔", Emotion_sadness_mean, "%")
    # print("화남", Emotion_angry_mean, "%")
    # print("중립", Emotion_neutral_mean, "%")
    # summmmm = Emotion_surprise_mean + Emotion_fear_mean + Emotion_disgust_mean + Emotion_happy_mean + Emotion_sadness_mean + Emotion_angry_mean + Emotion_neutral_mean
    # print('합', summmmm)

    # 시선 분석 결과
    # print(len(Gaze_list))

    # 얼굴 각도 결과
    Roll_mean_value = shoulder_Detector.Roll_slope_mean(Roll_list)
    # print(Roll_mean_value)

    # 어깨 각도 결과
    Shoulder_slope_mean_value = shoulder_Detector.Shoulder_slope_mean(Shoulder_slope_list)
    # print(Shoulder_slope_mean_value)

    # 어깨 움직임 결과
    # print("왼쪽어", (Left_shoulder_list))
    Left_shoulder_max = shoulder_calculate.Left_shoulder_max(Left_shoulder_list)
    # print(Left_shoulder_max)

    Left_shoulder_min = shoulder_calculate.Left_shoulder_min(Left_shoulder_list)
    # print(Left_shoulder_min)

    shoulder_left_count = Average.shoulder_left_count(shoulder_vertically_left_count)

    # print("오른쪽어", Right_shoulder_list)
    Right_shoulder_max = shoulder_calculate.Right_shoulder_max(Right_shoulder_list)
    # print(Right_shoulder_max)

    Right_shoulder_min = shoulder_calculate.Right_shoulder_min(Right_shoulder_list)
    # print(Right_shoulder_min)

    Right_left_count = Average.shoulder_right_count(shoulder_vertically_left_count)

    # print("가운데어", Center_shoulder_list)
    Center_shoulder_max = shoulder_calculate.Center_shoulder_max(Center_shoulder_list)
    # print(Center_shoulder_max)

    Center_shoulder_min = shoulder_calculate.Center_shoulder_min(Center_shoulder_list)
    # print(Center_shoulder_min)

    # 손
    Left_Hand_time = Left_Hand_time_calculation(Left_Hand_point_result)
    Right_Hand_time = Right_Hand_time_calculation(Right_Hand_point_result)


    # result_data = OrderedDict()
    # result_data["userkey"] = userkey
    # result_data["videoNo"] = videoNo
    # result_data["result"] = {"face_check": Face_analy_result, "sound_check": sound_confirm,
    #                          "emotion_surprise": Emotion_surprise_mean, "emotion_fear": Emotion_fear_mean,
    #                          "emotion_aversion": Emotion_disgust_mean, "emotion_happy": Emotion_happy_mean,
    #                          "emotion_sadness": Emotion_sadness_mean,
    #                          "emotion_angry": Emotion_angry_mean,
    #                          "emotion_neutral": Emotion_neutral_mean,
    #                          "gaze": Gaze_list, "face_angle": Roll_mean_value,
    #                          "shoulder_angle": Shoulder_slope_mean_value,
    #                          "left_shoulder": {"high_spot": Left_shoulder_max,
    #                                            "low_spot": Left_shoulder_min,
    #                                            "move_count": shoulder_vertically_left_count},
    #                          "right_shoulder": {"high_spot": Right_shoulder_max,
    #                                             "low_spot": Right_shoulder_min,
    #                                             "move_count": shoulder_vertically_right_count},
    #                          "center_shoulder": {"left_spot": Center_shoulder_max,
    #                                              "right_spot": Center_shoulder_min,
    #                                              "left_move_count": shoulder_horizontality_count_value[0],
    #                                              "right_move_count": shoulder_horizontality_count_value[1]},
    #                          "left_hand": {"time": Left_Hand_time, "count": Left_Hand_count,
    #                                        "point": Left_Hand_point_result},
    #                          "right_hand": {"time": Right_Hand_time, "count": Right_Hand_count,
    #                                         "point": Right_Hand_point_result}}

    # 점수화_표준편차
    Gaze_value = Average.Gaze_Avg(Gaze_list)
    Roll_value = Roll_mean_value
    Shoulder_velue = Shoulder_slope_mean_value
    vertically_value = Average.vertically_Avg(Left_shoulder_max,
                                              Left_shoulder_min,
                                              Right_shoulder_max,
                                              Right_shoulder_min)
    horizontally_value = Average.horizontally_Avg(Center_shoulder_max, Center_shoulder_min)
    GestureTIME_value = Average.GestureTIME(Left_Hand_time, Right_Hand_time)

    print('Left_shoulder_max >>> ', Left_shoulder_max)
    print('Right_shoulder_max >>> ', Right_shoulder_max)
    print('Center_shoulder_min >>> ', Center_shoulder_min)
    print('Left_shoulder >>>', Left_shoulder_list)



    gaze_dict = {"point": Gaze_list}
    left_shoulder_dict = {"high_spot": {"x": Left_shoulder_max[0], "y": Left_shoulder_max[1]},
                          "low_spot": {"x": Left_shoulder_min[0], "y": Left_shoulder_min[1]}}
    right_shoulder_dict = {"high_spot": {"x": Right_shoulder_max[0], "y": Right_shoulder_max[1]},
                           "low_spot": {"x": Right_shoulder_min[0], "y": Right_shoulder_min[1]}}
    center_shoulder_dict = {"left_spot": {"x": Center_shoulder_min[0], "y": Center_shoulder_min[1]},
                            "right_spot": {"x": Center_shoulder_max[0], "y": Center_shoulder_max[1]}}
    left_hand_dict = {"point": Left_Hand_point_result}
    right_hand_dict = {"point": Right_Hand_point_result}



    test = TestTable()
    test.USER_KEY = userkey
    test.QZ_NUM = videoNo
    test.FACE_CHECK = Face_analy_result
    test.SOUND_CHECK = sound_confirm
    test.EMOTION_SURPRISE = round(Emotion_surprise_mean, 5)
    test.EMOTION_FEAR = round(Emotion_fear_mean, 5)
    test.EMOTION_AVERSION = round(Emotion_disgust_mean, 5)
    test.EMOTION_HAPPY = round(Emotion_happy_mean, 5)
    test.EMOTION_SADNESS = round(Emotion_sadness_mean, 5)
    test.EMOTION_ANGRY = round(Emotion_angry_mean, 5)
    test.EMOTION_NEUTRAL = round(Emotion_neutral_mean, 5)
    test.GAZE = json.dumps(gaze_dict)
    test.FACE_ANGLE = round(Roll_mean_value, 5)
    test.SHOULDER_ANGLE = round(Shoulder_slope_mean_value, 5)
    test.LEFT_SHOULDER = json.dumps(left_shoulder_dict)
    test.LEFT_SHOULDER_MOVE_COUNT = shoulder_left_count
    test.RIGHT_SHOULDER = json.dumps(right_shoulder_dict)
    test.RIGHT_SHOULDER_MOVE_COUNT = Right_left_count
    test.CENTER_SHOULDER = json.dumps(center_shoulder_dict)
    test.CENTER_SHOULDER_LEFT_MOVE_COUNT = shoulder_horizontality_count_value[0]
    test.CENTER_SHOULDER_RIGHT_MOVE_COUNT = shoulder_horizontality_count_value[1]
    test.LEFT_HAND = json.dumps(left_hand_dict)
    test.LEFT_HAND_TIME = Left_Hand_time
    test.LEFT_HAND_MOVE_COUNT = Left_Hand_count
    test.RIGHT_HAND = json.dumps(right_hand_dict)
    test.RIGHT_HAND_TIME = Right_Hand_time
    test.RIGHT_HAND_MOVE_COUNT = Right_Hand_count

    test.GAZE_X_SCORE = scoring.GAZE_X_scoring(Gaze_value[0])
    test.GAZE_Y_SCORE = scoring.GAZE_Y_scoring(Gaze_value[1])
    test.SHOULDER_VERTICAL_SCORE = scoring.SHOULDER_VERTICAL_scoring(vertically_value)
    test.SHOULDER_HORIZON_SCORE = scoring.SHOULDER_HORIZON_scoring(horizontally_value)
    test.FACE_ANGLE_SCORE = scoring.FACE_ANGLE_scoring(Roll_value)
    test.GESTURE_SCORE = scoring.SHOULDER_ANGLE_scoring(Shoulder_slope_mean_value)


    # test.left_shoulder = {"high_spot": {"x":Left_shoulder_max[0], "y":Left_shoulder_max[1]}, "low_spot": {"x":Left_shoulder_min[0], "y":Left_shoulder_min[1]},
    #                                            "move_count": Left_shoulder_move_count}
    # test.right_shoulder = {"high_spot": {"x":Right_shoulder_max[0], "y":Right_shoulder_max[1]}, "low_spot": {"x":Right_shoulder_min[0], "y":Right_shoulder_min[1]},
    #                                             "move_count": Right_shoulder_move_count}
    # test.center_shoulder = {"left_spot": {"x":Center_shoulder_min[0], "y":Center_shoulder_min[1]}, "right_spot": {"x":Center_shoulder_max[0], "y":Center_shoulder_max[1]},
    #                                              "left_move_count": Center_shoulder_leftmove_count,
    #                                              "right_move_count": Center_shoulder_rightmove_count}
    # test.left_hand = {"time": Left_Hand_time, "count": Left_Hand_count,
    #                                        "point": Left_Hand_point_result}
    # test.right_hand = {"time": Right_Hand_time, "count": Right_Hand_count,
    #                                         "point": Right_Hand_point_result}




    session.add(test)
    session.commit()


class Item(BaseModel):
    userkey: int
    videoNo: int
    videoaddress: str

@app.post("/", status_code=202)
async def analy(item:Item, background_tasks: BackgroundTasks):
    userkey = item.userkey
    videoNo = item.videoNo
    videoaddress = item.videoaddress

    # vc = cv2.VideoCapture(videoaddress)
    #
    # test = True
    #
    # if vc.isOpened() == False:
    #     result = False
    #     return test
    #
    # # soundconfirm = soundcheck(videoaddress)
    # while(test):
    #     ret, frame = vc.read()
    #     print("1")
    #     if ret:
    #         if(int(vc.get(1)) % 5 == 0):
    #             frame = cv2.flip(frame, 1)
    #             img = frame
    #             img_show = copy.deepcopy(img)
    #             list_Face = list()
    #             # await Face_Detection(FD_Net, img, list_Face)
    #
    #             # background_tasks.add_task(Face_Detection, FD_Net, Landmark_Net, Headpose_Net, Emotion_Net, img, list_Face)

    # background_tasks.add_task(asyncio.run(video_task(userkey, videoNo, videoaddress)))
    background_tasks.add_task(video_task, userkey, videoNo, videoaddress)

        # else:
    return "1"


if __name__ == '__main__':
    uvicorn.run(app, port=8000)