import pyodbc

server = '172.11.12.93'  # IP máy chứa SQL Server
database = 'audio'
username = 'loctruong'
password = '11012004'
port = '1433'

conn_str = f'DRIVER={{SQL Server}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password}'
import os
import re
def correct_drive_path(path):
    match = re.match(r'^([A-Za-z])[\\/](.+)', path)
    if match:
        drive = match.group(1).upper()
        sub_path = match.group(2)
        fixed_path = f"{drive}:/{sub_path}"
        return os.path.normpath(fixed_path)
    return os.path.normpath(path)

def get_db(limit="TOP 10"):
    try:
        cnxn = pyodbc.connect(conn_str)
        path_save = []
        cursor = cnxn.cursor()
        cursor.execute(f'''
                            SELECT {limit} id, Path
                            FROM songs
                            WHERE Path IS NOT NULL AND type_music IS NULL
                            ORDER BY id ASC
                        ''')
        for row in cursor:
            # Chuyển đổi path ngay khi thêm vào list
            fixed_path = correct_drive_path(row[1])
            path_save.append({"id": int(row[0]), "path": fixed_path})
        cnxn.close()
        return path_save
    except Exception as e:
        print("Lỗi khi kết nối:", e)
        return []
import time

def execute_with_retry(query_func, retries=3, delay=1):
    for attempt in range(retries):
        try:
            query_func()
            return True
        except pyodbc.Error as e:
            if "1205" in str(e):  # Deadlock
                print(f"⚠ Deadlock! Thử lại {attempt+1}/{retries}...")
                time.sleep(delay)
            else:
                raise
    return False

def save_segment_prediction_to_db(song_id, segment_time, predicted_class, confidence):
    def run_query():
        try:
            cnxn = pyodbc.connect(conn_str)
            cursor = cnxn.cursor()

            # 1. Lấy toàn bộ dữ liệu từ bảng songs
            cursor.execute("SELECT id, Name, UPC, ISRC, F5, Path FROM songs WHERE id = ?", song_id)
            row = cursor.fetchone()
            if not row:
                print(f" Không tìm thấy bài hát với ID = {song_id}")
                cnxn.close()
                return

            # 2. Insert đầy đủ dữ liệu vào song_segments
            insert_query = """
                INSERT INTO song_segments (
                    id, Name, UPC, ISRC, F5, Path,
                    time, type_music, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(
                insert_query,
                row.id,
                row.Name,
                row.UPC,
                row.ISRC,
                row.F5,
                row.Path,
                segment_time,
                predicted_class,
                round(confidence * 100, 2)
            )

            # 3. Cập nhật bảng songs để đánh dấu đã dự đoán
            update_query = """
                UPDATE songs
                SET type_music = 1
                WHERE id = ?
            """
            cursor.execute(update_query, song_id)

            cnxn.commit()
            cnxn.close()
        except Exception as e:
            print(f"❌ Lỗi khi lưu segment ID={song_id}: {e}")
            print("⏳ Đợi 2 phút trước khi thử lại...")
            time.sleep(5)  # đợi 2 phút
            raise e  
    if execute_with_retry(run_query, retries=5, delay=0.5):
        print(f"✅ Lưu đầy đủ + cập nhật songs: {predicted_class} ({confidence*100:.2f}%) tại {segment_time}s cho ID = {song_id}")
    else:
        print(f"❌ Bỏ qua segment: {predicted_class} tại {segment_time}s cho ID = {song_id}")



