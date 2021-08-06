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

    # 사람 명수 체크 리스트
    Face_count_list = []
    # 분석 여부(얼굴 없으면)
    Face_analy_result = 0

    # 감정 담을 리스트
    Emotion_list = []
    # 시선 담을 리스트
    Gaze_list = []
    # 얼굴 각도 담을 리스트
    Roll_list = []
    # 어꺠 각도 담을 리스트
    Shoulder_slope_list = []
    # 왼쪽 어꺠 좌표 리스트
    Left_shoulder_list = []
    # 오른쪽 어깨 좌표 리스트
    Right_shoulder_list = []
    # 좌우 어깨 중간값 리스트
    Center_shoulder_list = []
    # 손 체크 리스트
    # Hand_list = []
    # Hand_count = 0
    # Hand_time_list = []
    # Hand_point_list = []
    # Hand_point_result = []

    # 손 체크 리스트 2222
    Left_Hand_list = []
    Left_Hand_count = 0
    Left_Hand_time_list = []
    Left_Hand_point_list = []
    Left_Hand_point_result = []

    Right_Hand_list = []
    Right_Hand_count = 0
    Right_Hand_time_list = []
    Right_Hand_point_list = []
    Right_Hand_point_result = []

    # 랜드마크 리스트(7번) / 어깨 위아래 움직임 체크 기준
    landmark_no7_y = 0
    # 랜드마크 리스트(1번, 17번) / 어깨 좌우 움직임 체크 기준
    landmark_no1_x = 0
    landmark_no17_x = 0
    # 왼쪽 어깨 움직임(위아래)
    Left_shoulder_move_list = []
    Left_shoulder_move_count = 0
    # 오른쪽 어깨 움직임(위아래)
    Right_shoulder_move_list = []
    Right_shoulder_move_count = 0

    # 어깨 좌우 움직임
    Center_shoulder_leftmove_list = []
    Center_shoulder_leftmove_count = 0

    Center_shoulder_rightmove_list = []
    Center_shoulder_rightmove_count = 0

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
                    landmark = Landmark_Detection(Landmark_Net, img, list_Face, 0)
                    if landmark_no7_y is 0:
                        landmark_no7_y = landmark[13]
                    if landmark_no1_x is 0:
                        landmark_no1_x = landmark[0]
                    if landmark_no17_x is 0:
                        landmark_no17_x = landmark[32]
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
                    print(Emotion_list)

                    # gaze regression
                    gaze = Gaze_Regression(list_Face, 0)

                    if gaze != None:
                        # print("gaze", gaze)
                        center_gaze_x = (gaze[0][0] + gaze[1][0]) / 2
                        center_gaze_y = (gaze[0][1] + gaze[1][1]) / 2
                        cv2.circle(img_show, (int(center_gaze_x), int(center_gaze_y)), 8, (0, 0, 255), -1)
                        center_gaze = (int(center_gaze_x), int(center_gaze_y))
                        Gaze_list.append(center_gaze)

                    # 손 추적
                    # img_show = hand_detector.findHands(img_show)
                    # lmList_hand = hand_detector.find_Hand_Position(img_show)
                    # if len(lmList_hand) != 0:
                    #     Hand_list.append(1)
                    #     Hand_time_list.append(1)
                    #     Hand_point_list.append(lmList_hand(12))
                    # else:
                    #     if len(Hand_list) > 3:
                    #         Hand_count += 1
                    #         print("손 계속계속", Hand_count)
                    #         Hand_point_result.append(Hand_point_list)
                    #     Hand_list = []
                    #     Hand_point_list = []

                    # if len(lmList_hand) != 0:
                    #     print("손", lmList_hand[4])

                    # 어깨 추적
                    pose_detector.findPose(img_show)

                    lmList_pose = pose_detector.findPosition(img_show)
                    # print("body", lmList_pose)

                    # 왼손 추적 22222222
                    if lmList_pose != 0:
                        if lmList_pose[15][1] < 640 and lmList_pose[15][2] < 480:
                            Left_hand = (lmList_pose[15][1], lmList_pose[15][2])
                            Left_Hand_list.append(1)
                            Left_Hand_time_list.append(1)
                            Left_Hand_point_list.append(Left_hand)
                        else:
                            if len(Left_Hand_list) > 3:
                                Left_Hand_count += 1
                                Left_Hand_point_result.append(Left_Hand_point_list)
                            Left_Hand_list = []
                            Left_Hand_point_list = []

                        # 오른손 추적 22222222
                        if lmList_pose[16][1] < 640 and lmList_pose[16][2] < 480:
                            Right_hand = (lmList_pose[16][1], lmList_pose[16][2])
                            Right_Hand_list.append(1)
                            Right_Hand_time_list.append(1)
                            Right_Hand_point_list.append(Right_hand)
                        else:
                            if len(Right_Hand_list) > 3:
                                Right_Hand_count += 1
                                Right_Hand_point_result.append(Right_Hand_point_list)
                            Right_Hand_list = []
                            Right_Hand_point_list = []

                        # 어깨 추적
                        # if len(lmList_pose) > 13:
                        #     cv2.circle(img_show, (lmList_pose[12][1], lmList_pose[12][2]), 8, (255, 255, 0), -1)
                        #     cv2.circle(img_show, (lmList_pose[11][1], lmList_pose[11][2]), 8, (255, 255, 0), -1)

                        left_shoulder = (lmList_pose[11][1], lmList_pose[11][2])
                        right_shoulder = (lmList_pose[12][1], lmList_pose[12][2])
                        center_shoulder = (int((lmList_pose[11][1] + lmList_pose[12][1]) / 2),
                                           int((lmList_pose[11][2] + lmList_pose[12][2]) / 2))

                        Left_shoulder_list.append(left_shoulder)
                        Right_shoulder_list.append(right_shoulder)
                        Center_shoulder_list.append(center_shoulder)

                        # 어꺠움직임 count
                        if left_shoulder[1] >= landmark_no7_y:
                            Left_shoulder_move_list.append(left_shoulder[1])
                        else:
                            if len(Left_shoulder_move_list) > 3:
                                Left_shoulder_move_count += 1
                                # print('1111111')
                            Left_shoulder_move_list = []

                        if right_shoulder[1] >= landmark_no7_y:
                            Right_shoulder_move_list.append(right_shoulder[1])
                        else:
                            if len(Right_shoulder_move_list) > 3:
                                Right_shoulder_move_count += 1
                                # print('2222222')
                            Right_shoulder_move_list = []

                        # 어깨 좌우 움직임 count
                        if center_shoulder[0] <= landmark_no1_x:
                            Center_shoulder_leftmove_list.append(center_shoulder[0])
                        else:
                            if len(Center_shoulder_leftmove_list) > 3:
                                Center_shoulder_leftmove_count += 1
                                # print('333333')
                            Center_shoulder_leftmove_list = []

                        if center_shoulder[0] >= landmark_no17_x:
                            Center_shoulder_rightmove_list.append(center_shoulder[0])
                        else:
                            if len(Center_shoulder_rightmove_list) > 3:
                                Center_shoulder_rightmove_count += 1
                                # print('4444')
                            Center_shoulder_rightmove_list = []

                        # 어깨 기울기
                        if right_shoulder[0] != left_shoulder[0]:
                            shoulder_slope = (right_shoulder[1] - left_shoulder[1]) / (
                                    right_shoulder[0] - left_shoulder[0])
                            Shoulder_slope_list.append(shoulder_slope)
        else:
            break

    Face_count_no_one = len(Face_count_list) - Face_count_list.count(1)
    # print(Face_count_no_one)
    if Face_count_no_one * 5 >= (FPS * 7):
        # print("분석 ㄴㄴ")
        Face_analy_result = False
    else:
        # print("분석 ok")
        Face_analy_result = True

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
    # print(len(Roll_list))
    Roll_sum = 0
    for ii in range(len(Roll_list)):
        Roll_sum = Roll_sum + Roll_list[ii]
    Roll_mean = Roll_sum / len(Roll_list)
    # print("얼굴 각도 평균", Roll_mean)

    # 어깨 각도 결과
    # print(len(Shoulder_slope_list))
    Shoulder_slope_sum = 0
    for iii in range(len(Shoulder_slope_list)):
        Shoulder_slope_sum = Shoulder_slope_sum + Shoulder_slope_list[iii]
    Shoulder_slope_mean = Shoulder_slope_sum / len(Shoulder_slope_list)
    # print("어깨 각도 평균", Shoulder_slope_mean)

    # 어깨 움직임 결과
    # print("왼쪽어", (Left_shoulder_list))
    Left_shoulder_max_y = max(t[1] for t in Left_shoulder_list)
    for x, y in enumerate(Left_shoulder_list):
        if Left_shoulder_max_y in y:
            Left_shoulder_max = y
    # print(Left_shoulder_max)

    Left_shoulder_min_y = min(t[1] for t in Left_shoulder_list)
    for x, y in enumerate(Left_shoulder_list):
        if Left_shoulder_min_y in y:
            Left_shoulder_min = y
    # print(Left_shoulder_min)

    # print("오른쪽어", Right_shoulder_list)
    Right_shoulder_max_y = max(t[1] for t in Right_shoulder_list)
    for x, y in enumerate(Right_shoulder_list):
        if Right_shoulder_max_y in y:
            Right_shoulder_max = y
    # print(Right_shoulder_max)

    Right_shoulder_min_y = min(t[1] for t in Right_shoulder_list)
    for x, y in enumerate(Right_shoulder_list):
        if Right_shoulder_min_y in y:
            Right_shoulder_min = y
    # print(Right_shoulder_min)

    # print("가운데어", Center_shoulder_list)
    Center_shoulder_max_x = max(t[0] for t in Center_shoulder_list)
    for x, y in enumerate(Center_shoulder_list):
        if Center_shoulder_max_x in y:
            Center_shoulder_max = y
    # print(Center_shoulder_max)

    Center_shoulder_min_x = min(t[0] for t in Center_shoulder_list)
    for x, y in enumerate(Center_shoulder_list):
        if Center_shoulder_min_x in y:
            Center_shoulder_min = y
    # print(Center_shoulder_min)

    # 손
    # print("손", Hand_list)
    # print(len(Hand_list))
    # print(Hand_count)
    # Hand_time = float(len(Hand_time_list) / 3)
    # print("손 등장 시간", Hand_time)
    # print("손 좌표", Hand_point_result)

    # 손222222
    # print("왼손 횟수", Left_Hand_count)
    Left_Hand_time = float(len(Left_Hand_time_list) / 6)
    # print("왼손 시간", Left_Hand_time)
    # print("왼손 좌표", Left_Hand_point_result)
    #
    # print("오른손 횟수", Right_Hand_count)
    Right_Hand_time = float(len(Right_Hand_time_list) / 6)
    # print("오른손 시간", Right_Hand_time)
    # print("오른손 좌표", Right_Hand_point_result)
    #
    #
    # # 어깨 움직임 체크 위아래
    # print("왼쪽 어깨 움직임", Left_shoulder_move_count)
    # print("오른쪽 어깨 움직임", Right_shoulder_move_count)
    #
    # # 어깨 움직임 체크 좌우
    # print("왼쪽방향으로 움직임", Center_shoulder_leftmove_count)
    # print("오른쪽방향으로 움직임", Center_shoulder_rightmove_count)
    # # print("Done")
    #
    # print(Face_analy_result)

    # json 저장

    result_data = OrderedDict()

    result_data["userkey"] = userkey
    result_data["videoNo"] = videoNo
    result_data["result"] = {"face_check": Face_analy_result, "sound_check": sound_confirm,
                             "emotion_surprise": Emotion_surprise_mean, "emotion_fear": Emotion_fear_mean,
                             "emotion_aversion": Emotion_disgust_mean, "emotion_happy": Emotion_happy_mean,
                             "emotion_sadness": Emotion_sadness_mean,
                             "emotion_angry": Emotion_angry_mean, "emotion_neutral": Emotion_neutral_mean,
                             "gaze": Gaze_list, "face_angle": Roll_mean, "shoulder_angle": Shoulder_slope_mean,
                             "left_shoulder": {"high_spot": Left_shoulder_max, "low_spot": Left_shoulder_min,
                                               "move_count": Left_shoulder_move_count},
                             "right_shoulder": {"high_spot": Right_shoulder_max, "low_spot": Right_shoulder_min,
                                                "move_count": Right_shoulder_move_count},
                             "center_shoulder": {"left_spot": Center_shoulder_min, "right_spot": Center_shoulder_max,
                                                 "left_move_count": Center_shoulder_leftmove_count,
                                                 "right_move_count": Center_shoulder_rightmove_count},
                             "left_hand": {"time": Left_Hand_time, "count": Left_Hand_count,
                                           "point": Left_Hand_point_result},
                             "right_hand": {"time": Right_Hand_time, "count": Right_Hand_count,
                                            "point": Right_Hand_point_result}}

    # with open('/home/ubuntu/projects/withmind_video/im_video/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w', encoding='utf-8') as make_file:
    #      json.dump(result_data, make_file, ensure_ascii=False, indent="\t")

    with open('C:/Users/withmind/Desktop/models/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w',
              encoding='utf-8') as make_file:
        json.dump(result_data, make_file, ensure_ascii=False, indent="\t")

    ftp = FTP()

    ftp.connect('withmind.cache.smilecdn.com', 21)
    ftp.login('withmind', 'dnlemakdlsem1!')
    ftp.cwd('./analy_result')
    filename = '%d_%d_result.json' % (int(userkey), int(videoNo))
    # myfile = open('C:/Users/withmind/Documents/withmind/IM/01_python_pytorch_project/01_python_pytorch_project/video/im_video/' + filename, 'wb')
    # with open('C:/Users/withmind/Documents/withmind/IM/01_python_pytorch_project/01_python_pytorch_project/video/im_video/' + filename) as contents:
    #     ftp.storbinary('STOR %s' % filename, contents)
    # fileroute = '/home/ubuntu/projects/withmind_video/im_video/'
    fileroute = 'C:/Users/withmind/Desktop/models/'
    myfile = open(fileroute + filename, 'rb')
    ftp.storbinary('STOR ' + filename, myfile)

    myfile.close()

    os.remove(fileroute + filename)

    test = TestTable()
    test.userkey = userkey
    test.videoNo = videoNo
    test.face_check = Face_analy_result
    test.sound_check = sound_confirm
    test.emotion_surprise = Emotion_surprise_mean
    test.emotion_fear = Emotion_fear_mean
    test.emotion_aversion = Emotion_disgust_mean
    test.emotion_happy = Emotion_happy_mean
    test.emotion_sadness = Emotion_sadness_mean
    test.emotion_angry = Emotion_angry_mean
    test.emotion_neutral = Emotion_neutral_mean
    test.gaze = Gaze_list
    test.face_angle = Roll_mean
    test.shoulder_angle = Shoulder_slope_mean
    test.left_shoulder = {"high_spot": Left_shoulder_max, "low_spot": Left_shoulder_min,
                                               "move_count": Left_shoulder_move_count}
    test.right_shoulder = {"high_spot": Right_shoulder_max, "low_spot": Right_shoulder_min,
                                                "move_count": Right_shoulder_move_count}
    test.center_shoulder = {"left_spot": Center_shoulder_min, "right_spot": Center_shoulder_max,
                                                 "left_move_count": Center_shoulder_leftmove_count,
                                                 "right_move_count": Center_shoulder_rightmove_count}
    test.left_hand = {"time": Left_Hand_time, "count": Left_Hand_count,
                                           "point": Left_Hand_point_result}
    test.right_hand = {"time": Right_Hand_time, "count": Right_Hand_count,
                                            "point": Right_Hand_point_result}




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

    background_tasks.add_task(video_task, userkey, videoNo, videoaddress)

        # else:
    return "1"


if __name__ == '__main__':
    uvicorn.run(app, port=8000)