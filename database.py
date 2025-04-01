import sqlite3
from contextlib import contextmanager

class Database:
    def __init__(self, db_name='tasks.db'):
        self.db_name = db_name
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            # create Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Users (
                    UserID INTEGER PRIMARY KEY AUTOINCREMENT,
                    Username TEXT UNIQUE NOT NULL,
                    Email TEXT UNIQUE NOT NULL,
                    PasswordHash TEXT NOT NULL,
                    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # create Tasks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Tasks (
                    TaskID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID INTEGER,
                    TaskDesc TEXT NOT NULL,
                    Date DATE,
                    Time TIME,
                    Category TEXT,
                    Urgency INTEGER,
                    Status TEXT DEFAULT 'Pending',
                    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(UserID) REFERENCES Users(UserID)
                )
            ''')

    def create_task(self, user_id, task_desc, **kwargs):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Tasks 
                (UserID, TaskDesc, Date, Time, Category, Urgency, Status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                task_desc,
                kwargs.get('Date'),
                kwargs.get('Time'),
                kwargs.get('Category'),
                kwargs.get('Urgency'),
                kwargs.get('Status')
            ))
            conn.commit()