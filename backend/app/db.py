import psycopg2

from app.config import settings


def get_connection():
    return psycopg2.connect(settings.supabase_db_url)
