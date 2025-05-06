import sqlite3
from contextlib import contextmanager

class Database:
    def __init__(self, db_name='tasks.db'):
        self.db_name = db_name
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.execute("PRAGMA foreign_keys = ON")
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
            
            # create Tasks table #### add status (pending/done)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS Tasks (
                    TaskID INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserID INTEGER,
                    TaskDesc TEXT NOT NULL,
                    Date DATE,
                    Time TIME,
                    Category TEXT,
                    
                    Status TEXT DEFAULT 'pending',
                    CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UpdatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(UserID) REFERENCES Users(UserID)
                )
            ''')

    def create_task(self, user_id, **kwargs):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Tasks
                (UserID, TaskDesc, Date, Time, Category , Status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                kwargs.get('TaskDesc'),
                kwargs.get('Date'),
                kwargs.get('Time'),
                kwargs.get('Category'),
                
                'pending' # default status
            ))
            conn.commit()

    def get_tasks(self, user_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TaskID, TaskDesc, Date, Time, Category , Status FROM Tasks WHERE UserID = ?", (user_id,))
            return cursor.fetchall()
    
    def update_task_status(self, task_id, status):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE Tasks SET Status = ? WHERE TaskID = ?", (status, task_id))
            conn.commit()
    
    def update_task(self, task_id, **kwargs):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            update_fields = []
            update_values = []
            for key, value in kwargs.items():
                if key != 'TaskID':
                    update_fields.append(f"{key} = ?")
                    update_values.append(value)
            update_values.append(task_id)
            update_query = f"UPDATE Tasks SET {', '.join(update_fields)} WHERE TaskID = ?"
            cursor.execute(update_query, update_values)
            conn.commit()
            
    def delete_task(self, task_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM Tasks WHERE TaskID = ?", (task_id,))
            conn.commit()

    def get_user_by_email(self, email):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT UserID, Username, PasswordHash 
                FROM Users 
                WHERE LOWER(TRIM(Email)) = LOWER(TRIM(?))
            """, (email,))
            return cursor.fetchone()

    def get_task(self, task_id):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TaskID, UserID, TaskDesc, Date, Time, Category , Status 
                FROM Tasks 
                WHERE TaskID = ?
            """, (task_id,))
            return cursor.fetchone()

    def create_user(self, username, email, password_hash):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Users (Username, Email, PasswordHash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            conn.commit()
            return cursor.lastrowid  # Returns the new UserID

    def get_user_by_username(self, username):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Use LOWER() for case-insensitive comparison and TRIM() for whitespace
            cursor.execute("""
                SELECT UserID, Username, PasswordHash 
                FROM Users 
                WHERE LOWER(TRIM(Username)) = LOWER(TRIM(?))
            """, (username,))
            result = cursor.fetchone()
            print(f"DEBUG: User lookup for '{username}' returned: {result}")  # Debug print
            return result
        
    def debug_print_all_users(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Users")
            print("DEBUG: All users in database:")
            for row in cursor.fetchall():
                print(row)
