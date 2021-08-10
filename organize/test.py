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

    Gaze_list = []
    Roll_list = []
    Emotion_list = []
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

                    Landmark_list = Landmark_Detection(Landmark_Net, img, list_Face, 0)

                    # 이하 추가적인 인식은 첫번째 얼굴에 대해서만 수행.
                    # nIndex에 원하는 얼굴을 선택 가능

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
                        shoulder_vertically_left_count.append(shoulder_vertically_left(left_shoulder, Landmark_list))
                        shoulder_vertically_right_count.append(shoulder_vertically_right(right_shoulder, Landmark_list))

                        # 어깨 좌우
                        shoulder_horizontality_count_value = shoulder_horizontality_count(center_shoulder_left,
                                                                                          Landmark_list)

                        # 어깨 기울기
                        shoulder_slope_value = shoulder_slope(right_shoulder, left_shoulder)
                        Shoulder_slope_list.append(shoulder_slope_value)
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

    Emotion_analysi = Emotion_analysis(Emotion_list)

    # print("놀람", Emotion_analysi[0], "%")
    # print("공포", Emotion_analysi[1], "%")
    # print("역겨움", Emotion_analysi[2], "%")
    # print("행복", Emotion_analysi[3], "%")
    # print("슬픔", Emotion_analysi[4], "%")
    # print("화남", Emotion_analysi[5], "%")
    # print("중립", Emotion_analysi[6], "%")
    # print('합', Emotion_analysi[7])

    # 시선 분석 결과
    # print(len(Gaze_list))

    # 얼굴 각도 결과
    Roll_mean_value = Roll_mean(Roll_list)
    # print(Roll_mean_value)

    # 어깨 각도 결과
    Shoulder_slope_mean_value = Shoulder_slope_mean(Shoulder_slope_list)
    # print(Shoulder_slope_mean_value)

    # 어깨 움직임 결과
    # print("왼쪽어", (Left_shoulder_list))
    Left_shoulder_max_y_value = Left_shoulder_max_y(Left_shoulder_list)
    # print(Left_shoulder_max_y_value)

    Left_shoulder_min_y_value = Left_shoulder_min_y(Left_shoulder_list)
    # print(Left_shoulder_min_y_value)

    # print("오른쪽어", Right_shoulder_list)
    Right_shoulder_max_y_value = Right_shoulder_max_y(Right_shoulder_list)
    # print(Right_shoulder_max_y_value)

    Right_shoulder_min_y_value = Right_shoulder_min_y(Right_shoulder_list)
    # print(Right_shoulder_min_y_value)

    # print("가운데어", Center_shoulder_list)
    Center_shoulder_max_x_value = Center_shoulder_max_x(Center_shoulder_list)
    # print(Center_shoulder_max_x_value)

    Center_shoulder_min_x_value = Center_shoulder_min_x(Center_shoulder_list)
    # print(Center_shoulder_min_x_value)

    # 손
    Left_Hand_time = Left_Hand_time_calculation(Left_Hand_time_list)
    Right_Hand_time = Right_Hand_time_calculation(Right_Hand_time_list)

    result_data = OrderedDict()

    result_data["userkey"] = userkey
    result_data["videoNo"] = videoNo
    result_data["result"] = {"face_check": Face_analy_result, "sound_check": 'sound_confirm',
                             "emotion_surprise": Emotion_analysi[0], "emotion_fear": Emotion_analysi[1],
                             "emotion_aversion": Emotion_analysi[2], "emotion_happy": Emotion_analysi[3],
                             "emotion_sadness": Emotion_analysi[4],
                             "emotion_angry": Emotion_analysi[5], "emotion_neutral": Emotion_analysi[6],
                             "gaze": Gaze_list, "face_angle": Roll_mean_value,
                             "shoulder_angle": Shoulder_slope_mean_value,
                             "left_shoulder": {"high_spot": Left_shoulder_max_y_value,
                                               "low_spot": Left_shoulder_min_y_value,
                                               "move_count": shoulder_vertically_left_count},  # 왼쪽 위아래
                             "right_shoulder": {"high_spot": Right_shoulder_max_y_value,
                                                "low_spot": Right_shoulder_min_y_value,
                                                "move_count": shoulder_vertically_right_count},  # 오른쪽 위아래
                             "center_shoulder": {"left_spot": Center_shoulder_max_x_value,  # 가로 움직임
                                                 "right_spot": Center_shoulder_min_x_value,
                                                 "left_move_count": shoulder_horizontality_count_value[0],
                                                 "right_move_count": shoulder_horizontality_count_value[1]},
                             "left_hand": {"time": Left_Hand_time, "count": Left_Hand_count,
                                           "point": Left_Hand_point_result},
                             "right_hand": {"time": Right_Hand_time, "count": Right_Hand_count,
                                            "point": Right_Hand_point_result}}

    # with open('/home/ubuntu/projects/withmind_video/im_video/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w', encoding='utf-8') as make_file:
    #      json.dump(result_data, make_file, ensure_ascii=False, indent="\t")

    Gaze_velue = Average.Gaze_Avg(Gaze_list)
    Roll_velue = Roll_mean_value
    Shoulder_velue = Shoulder_slope_mean_value
    shoulder_left_count = shoulder_vertically_left_count
    shoulder_right_count = shoulder_vertically_right_count
    vertically_value = Average.vertically_Avg(Left_shoulder_max_y_value,
                                              Left_shoulder_min_y_value,
                                              Right_shoulder_max_y_value,
                                              Right_shoulder_min_y_value)
    horizontally_value = Average.horizontally_Avg(Center_shoulder_max_x_value, Center_shoulder_min_x_value)
    GestureTIME_value = Average.GestureTIME(Left_Hand_time, Right_Hand_time)

    # CSV 저장
    Average.Average_csv(Gaze_velue, Roll_velue, Shoulder_velue, shoulder_left_count, shoulder_right_count,
                        vertically_value, horizontally_value, GestureTIME_value)

    print('Gaze >> ', Gaze_velue, 'Roll >> ', Roll_velue,
          'Shoulder >> ', Shoulder_velue, 'shoulder_left_count >> ', shoulder_left_count, 'shoulder_right_count >> ',
          shoulder_right_count,
          'vertically >> ', vertically_value, 'horizontally >> ', horizontally_value, 'GestureTIME >> ',
          GestureTIME_value)


    with open('C:/Users/withmind/Desktop/models/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w', encoding='utf-8') as make_file:
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
    test.emotion_surprise = Emotion_analysi[0]
    test.emotion_fear = Emotion_analysi[1]
    test.emotion_aversion = Emotion_analysi[2]
    test.emotion_happy = Emotion_analysi[3]
    test.emotion_sadness = Emotion_analysi[4]
    test.emotion_angry = Emotion_analysi[5]
    test.emotion_neutral = Emotion_analysi[6]
    test.gaze = Gaze_list
    test.face_angle = Roll_mean
    test.shoulder_angle = Shoulder_slope_mean
    test.left_shoulder = {"high_spot": Left_shoulder_max_y_value, "low_spot": Left_shoulder_min_y_value,
                                               "move_count": shoulder_vertically_left_count}
    test.right_shoulder = {"high_spot": Right_shoulder_max_y_value, "low_spot": Right_shoulder_min_y_value,
                                                "move_count": shoulder_vertically_right_count}
    test.center_shoulder = {"left_spot": Center_shoulder_max_x_value,
                            "right_spot": Center_shoulder_min_x_value,
                            "left_move_count": shoulder_horizontality_count_value[0],
                            "right_move_count": shoulder_horizontality_count_value[1]}
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
async def analy(item: Item, background_tasks: BackgroundTasks):
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