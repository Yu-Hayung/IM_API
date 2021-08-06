from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session


user_name = "root"
user_pwd = "withmind1!"
db_host = "rdsimtest.ciravsrvnjpk.ap-northeast-2.rds.amazonaws.com"
db_name = "kangmin"


DATABASE = 'mysql+pymysql://%s:%s@%s/%s?charset=utf8' % (
    user_name,
    user_pwd,
    db_host,
    db_name,
)

ENGINE = create_engine(
    DATABASE,
    encoding='utf-8',
    echo=True
)

session = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=ENGINE
    )
)

Base = declarative_base()
Base.query = session.query_property()

# SQLALCHEMY_DATABASE_URL = "rdsimtest.ciravsrvnjpk.ap-northeast-2.rds.amazonaws.com/INTERVIEWMASTER"
#
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL
# )

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#
# Base = declarative_base()

