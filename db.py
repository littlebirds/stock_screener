from sqlalchemy import DATE, create_engine, Column, Integer, Double, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Index

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

class FundmentalInfo(Base):
    __tablename__ = "stock_info"
    ticker = Column(String(8), primary_key=True)
    info = Column(JSON, nullable=False)
    last_updated = Column(DATE) # last updated


# SQLite database connection
DB_FILE = "./stock_screener.db"
engine = create_engine(f"sqlite+pysqlite:///{DB_FILE}", echo=False)
Base.metadata.create_all(engine)

# Create a configured "Session" class
Session = sessionmaker(bind=engine)
# Create a Session
session = Session()

# Example of adding a new user
# new_user = User(username='johndoe', email='johndoe@example.com', password='securepassword')
# session.add(new_user)
# session.commit()
