from rest_framework.views import APIView
import copy
from .Face_Information import *
import json
from django.http import HttpResponse
from PIL import Image
from collections import OrderedDict
import os
from ftplib import FTP
import torch

import time

class IM_video_Anaylysis(APIView):
    def post(self, request):
        insert_data = json.loads(request.body)
        userkey = insert_data.get('userkey')
        videoNo = insert_data.get('videoNo')
        videoaddress = insert_data.get('videoaddress')

        print(userkey, videoNo, videoaddress)

        # model initialization
        FD_Net, Landmark_Net, Headpose_Net, Emotion_Net = Initialization()

        hand_detector = hand_Detector()
        pose_detector = pose_Detecor()

        # opencv cam initialization
        # vc = cv2.VideoCapture(url)
        vc = cv2.VideoCapture(videoaddress)
        # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        FPS = cv2.CAP_PROP_FPS
        sound_confirm = soundcheck(videoaddress)

        # 어깨 좌우 움직임
        bReady = True

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


        if vc.isOpened() == False:
            bReady = False
            print("Error : Cam is not opened.")

        start = time.time()
        while (bReady):
            # Frame read
            ret, frame = vc.read()
            if ret:
                if (int(vc.get(1)) % 5 == 0):
                    frame = cv2.flip(frame, 1)
                    img = frame
                    img_show = copy.deepcopy(img)

                    list_Face = []
                    Face_count_list = []

                    Face_Detection(FD_Net, img, list_Face)
                    Face_count_list.append(len(list_Face))

                    if len(list_Face) > 0:
                        # draw face ROI. list_ETRIFace 를 순회하며 모든 검출된 얼굴의 박스 그리기
                        for ii in range(len(list_Face)):
                            cv2.rectangle(img_show, (list_Face[ii].rt[0], list_Face[ii].rt[1])
                                          , (list_Face[ii].rt[2], list_Face[ii].rt[3]),
                                          (0, 255, 0), 2)

                        Landmark_list = Landmark_Detection(Landmark_Net, img, list_Face, 0)

                        # 얼굴 검출
                        pose = HeadPose_Estimation(Headpose_Net, img, list_Face, 0)
                        Roll_list.append(pose[2].item())

                        Emotion_Classification(Emotion_Net, img, list_Face, 0)
                        sEmotionLabel = ["surprise", "fear", "disgust", "happy", "sadness", "angry", "neutral"]
                        sEmotionResult = "Emotion : %s" % sEmotionLabel[list_Face[0].nEmotion]
                        EmotionResult = list_Face[0].fEmotionScore

                        Emotion_list.append(EmotionResult)

                        # 시선
                        gaze = Gaze_Regression(list_Face, 0)
                        Gaze_list.append(pose_Detecor.gaze_Detecor(gaze, img_show))

                        # 동작 검출
                        pose_detector.findPose(img_show)
                        lmList_pose = pose_detector.findPosition(img_show)

                        # 손
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
                        shoulder_horizontality_count_value = shoulder_horizontality_count(center_shoulder_left, Landmark_list)

                        # 어깨 기울기
                        shoulder_slope_value = shoulder_slope(right_shoulder, left_shoulder)
                        Shoulder_slope_list.append(shoulder_slope_value)



                # 결과 출력
                #cv2.imshow("T1", img_show)
                # nKey = cv2.waitKey(1)

                # esc로 프로그램 종료
                #if nKey == 27:
                #    break
            else:
                vc.release()
                #cv2.destroyAllWindows()
                break

        print(" 분석 시간(s) >> ", time.time() - start)


        Face_count_no_one = len(Face_count_list) - Face_count_list.count(1)
        # print(Face_count_no_one)
        if Face_count_no_one * 5 >= (FPS * 7):
            print("분석 ㄴㄴ")
            Face_analy_result = False
        else:
            print("분석 ok")
            Face_analy_result = True


        # Emotion_analysi = Emotion_analysis(Emotion_list)
        #
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
                                 "emotion_surprise": 'Emotion_analysi[0]', "emotion_fear": 'Emotion_analysi[1]',
                                 "emotion_aversion":'Emotion_analysi[2]', "emotion_happy": 'Emotion_analysi[3]',
                                 "emotion_sadness": 'Emotion_analysi[4]',
                                 "emotion_angry": 'Emotion_analysi[5]', "emotion_neutral": 'Emotion_analysi[6]',
                                 "gaze": Gaze_list, "face_angle": Roll_mean_value,
                                 "shoulder_angle": Shoulder_slope_mean_value,
                                 "left_shoulder": {"high_spot": Left_shoulder_max_y_value, "low_spot": Left_shoulder_min_y_value,
                                                   "move_count": shoulder_vertically_left_count},   #왼쪽 위아래
                                 "right_shoulder": {"high_spot": Right_shoulder_max_y_value, "low_spot": Right_shoulder_min_y_value,
                                                    "move_count": shoulder_vertically_right_count}, #오른쪽 위아래
                                 "center_shoulder": {"left_spot": Center_shoulder_max_x_value,   # 가로 움직임
                                                     "right_spot": Center_shoulder_min_x_value,
                                                     "left_move_count": shoulder_horizontality_count_value[0],
                                                     "right_move_count": shoulder_horizontality_count_value[1]},
                                 "left_hand": {"time": Left_Hand_time, "count": Left_Hand_count,
                                               "point": Left_Hand_point_result},
                                 "right_hand": {"time": Right_Hand_time, "count": Right_Hand_count,
                                                "point": Right_Hand_point_result}}

        # with open('/home/ubuntu/projects/withmind_video/im_video/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w', encoding='utf-8') as make_file:
        #      json.dump(result_data, make_file, ensure_ascii=False, indent="\t")

        # with open('C:/Users/yuhay/Desktop/Analysis_file/%d_%d_result.json' % (int(userkey), int(videoNo)), 'w', encoding='utf-8') as make_file:
        #      json.dump(result_data, make_file, ensure_ascii=False, indent="\t")

        # ftp_Analysis_json(userkey, videoNo)

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
        Average.Average_csv(Gaze_velue, Roll_velue, Shoulder_velue, shoulder_left_count, shoulder_right_count,vertically_value, horizontally_value, GestureTIME_value)
        print('Gaze >> ', Gaze_velue, 'Roll >> ', Roll_velue,
              'Shoulder >> ', Shoulder_velue, 'shoulder_left_count >> ', shoulder_left_count, 'shoulder_right_count >> ', shoulder_right_count,
              'vertically >> ', vertically_value, 'horizontally >> ', horizontally_value, 'GestureTIME >> ', GestureTIME_value)


        return HttpResponse("Done", status=200)
        # else:
        #     return Response('Fail', status=status.HTTP_200_OK)

