import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    
    X_API_KEY = os.environ.get('X_API_KEY') or "TWITTER API KEY NOT SET"
    X_BEARER_TOKEN = os.environ.get('X_BEARER_TOKEN') or "TWITTER BEARER TOKEN NOT SET"

    

