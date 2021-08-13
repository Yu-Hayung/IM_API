from sqlalchemy import Column, Integer, String, Float, Boolean
from pydantic import BaseModel
from database import Base
from database import ENGINE

class TestTable(Base):
    __tablename__ = 'IM_QZ_ANALYSIS'
    ANLYS_KEY = Column(Integer, primary_key=True, autoincrement=True)
    USER_KEY = Column(Integer, nullable=False)
    QZ_NUM = Column(Integer, nullable=False)
    FACE_CHECK = Column(String(10), nullable=False)
    SOUND_CHECK = Column(String(10), nullable=False)
    EMOTION_SURPRISE = Column(Float)
    EMOTION_FEAR = Column(Float)
    EMOTION_AVERSION = Column(Float)
    EMOTION_HAPPY = Column(Float)
    EMOTION_SADNESS = Column(Float)
    EMOTION_ANGRY = Column(Float)
    EMOTION_NEUTRAL = Column(Float)
    GAZE = Column(String(5000))
    FACE_ANGLE = Column(Float)
    SHOULDER_ANGLE = Column(Float)
    LEFT_SHOULDER = Column(String(5000))
    LEFT_SHOULDER_MOVE_COUNT = Column(Integer)
    RIGHT_SHOULDER = Column(String(5000))
    RIGHT_SHOULDER_MOVE_COUNT = Column(Integer)
    CENTER_SHOULDER = Column(String(5000))
    CENTER_SHOULDER_LEFT_MOVE_COUNT = Column(Integer)
    CENTER_SHOULDER_RIGHT_MOVE_COUNT = Column(Integer)
    LEFT_HAND = Column(String(5000))
    LEFT_HAND_TIME = Column(Integer)
    LEFT_HAND_MOVE_COUNT = Column(Integer)
    RIGHT_HAND = Column(String(5000))
    RIGHT_HAND_TIME = Column(Integer)
    RIGHT_HAND_MOVE_COUNT = Column(Integer)
    GAZE_X_SCORE = Column(Integer)
    GAZE_Y_SCORE = Column(Integer)
    SHOULDER_VERTICAL_SCORE = Column(Integer)
    SHOULDER_HORIZON_SCORE = Column(Integer)
    FACE_ANGLE_SCORE = Column(Integer)
    GESTURE_SCORE = Column(Integer)


class Test(BaseModel):
    ANLYS_KEY : int
    USER_KEY : int
    QZ_NUM : int
    FACE_CHECK : str
    SOUND_CHECK : str
    EMOTION_SURPRISE : float
    EMOTION_FEAR : float
    EMOTION_AVERSION : float
    EMOTION_HAPPY : float
    EMOTION_SADNESS : float
    EMOTION_ANGRY : float
    EMOTION_NEUTRAL : float
    GAZE : str
    FACE_ANGLE : float
    SHOULDER_ANGLE : float
    LEFT_SHOULDER : str
    LEFT_SHOULDER_MOVE_COUNT : int
    RIGHT_SHOULDER : str
    RIGHT_SHOULDER_MOVE_COUNT : int
    CENTER_SHOULDER : str
    CENTER_SHOULDER_LEFT_MOVE_COUNT : int
    CENTER_SHOULDER_RIGHT_MOVE_COUNT : int
    LEFT_HAND : str
    LEFT_HAND_TIME : int
    LEFT_HAND_MOVE_COUNT : int
    RIGHT_HAND : str
    RIGHT_HAND_TIME : int
    RIGHT_HAND_MOVE_COUNT : int
    GAZE_X_SCORE : int
    GAZE_Y_SCORE : int
    SHOULDER_VERTICAL_SCORE : int
    SHOULDER_HORIZON_SCORE : int
    FACE_ANGLE_SCORE : int
    GESTURE_SCORE : int