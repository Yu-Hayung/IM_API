from sqlalchemy import Column, Integer, String, Float, Boolean
from pydantic import BaseModel
from database import Base
from database import ENGINE

class TestTable(Base):
    __tablename__ = 'analy_test'
    id = Column(Integer, primary_key=True, autoincrement=True)
    userkey = Column(Integer, nullable=False)
    videoNo = Column(Integer, nullable=False)
    face_check = Column(Boolean, nullable=False)
    sound_check = Column(Boolean, nullable=False)
    emotion_surprise = Column(Float, nullable=False)
    emotion_fear = Column(Float, nullable=False)
    emotion_aversion = Column(Float, nullable=False)
    emotion_happy = Column(Float, nullable=False)
    emotion_sadness = Column(Float, nullable=False)
    emotion_angry = Column(Float, nullable=False)
    emotion_neutral = Column(Float, nullable=False)
    gaze = Column(String(5000), nullable=False)
    face_angle = Column(Float, nullable=False)
    shoulder_angle = Column(Float, nullable=False)
    left_shoulder = Column(String(5000), nullable=False)
    right_shoulder = Column(String(5000), nullable=False)
    center_shoulder = Column(String(5000), nullable=False)
    left_hand = Column(String(5000), nullable=False)
    right_hand = Column(String(5000), nullable=False)


class Test(BaseModel):
    id : int
    userkey : int
    videoNo : int
    face_check : bool
    sound_check : bool
    emotion_surprise : float
    emotion_fear : float
    emotion_aversion : float
    emotion_happy : float
    emotion_sadness : float
    emotion_angry : float
    emotion_neutral : float
    gaze : str
    face_angle : float
    shoulder_angle : float
    left_shoulder : str
    right_shoulder : str
    center_shoulder : str
    left_hand : str
    right_hand : str