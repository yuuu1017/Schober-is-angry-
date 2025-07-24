from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time
import os
from playsound import playsound

# --- 核心設定 ---
CAMERA_INDEX = 0
ANGRY_TEXT = "Schober is angry"
AUDIO_FILE = "annngry.mp3"  # <--- 在這裡修改音檔名稱
SOUND_COOLDOWN_SECONDS = 4  # 音效播放的冷卻時間（秒）
# --- 設定結束 ---

app = Flask(__name__)

# --- 檢查音檔是否存在 ---
if not os.path.exists(AUDIO_FILE):
    print(f"!!!!!!!!!!!!!!! 致命錯誤 !!!!!!!!!!!!!!!")
    print(f"找不到音檔: '{AUDIO_FILE}'")
    print(f"請確認您已經將一個名為 '{AUDIO_FILE}' 的 mp3 檔案放在與 app.py 同一個資料夾中。")
    input("按 Enter 鍵結束程式...")
    exit()

# 載入 YOLOv8n 模型
print(">>> 正在載入 YOLOv8 模型，請稍候...")
model = YOLO('yolov8n.pt')
print(">>> YOLOv8 模型載入完成！")

# 設定攝影機
print(f">>> 嘗試開啟索引值為 {CAMERA_INDEX} 的攝影機...")
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print(f"!!!!!!!!!!!!!!! 致命錯誤 !!!!!!!!!!!!!!!")
    print(f"無法開啟索引值為 {CAMERA_INDEX} 的攝影機！程式無法執行。")
    input("按 Enter 鍵結束程式...")
    exit()
else:
    print(f">>> 成功開啟攝影機！準備開始影像串流...")

# 全域變數，用來記錄上次播放音效的時間戳
last_sound_play_time = 0

def check_touch(person_box, phone_box, proximity=1.1):
    px_min, py_min, px_max, py_max = person_box
    phx_min, phy_min, phx_max, phy_max = phone_box
    person_width = px_max - px_min
    person_height = py_max - py_min
    px_min_expanded = px_min - (person_width * (proximity - 1) / 2)
    py_min_expanded = py_min - (person_height * (proximity - 1) / 2)
    px_max_expanded = px_max + (person_width * (proximity - 1) / 2)
    py_max_expanded = py_max + (person_height * (proximity - 1) / 2)
    phone_center_x = (phx_min + phx_max) / 2
    phone_center_y = (phy_min + phy_max) / 2
    if px_min_expanded < phone_center_x < px_max_expanded and py_min_expanded < phone_center_y < py_max_expanded:
        return True
    return False

def generate_frames():
    global last_sound_play_time

    while True:
        success, frame = camera.read()
        if not success:
            print("無法從攝影機讀取畫面，可能連線中斷。")
            time.sleep(1)
            continue

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        persons = []
        phones = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                c = int(box.cls[0])
                if model.names[c] == 'person':
                    persons.append(b)
                elif model.names[c] == 'cell phone':
                    phones.append(b)

        is_angry = False
        if persons and phones:
            for person_box in persons:
                for phone_box in phones:
                    if check_touch(person_box, phone_box):
                        is_angry = True
                        break
                if is_angry:
                    break
       
        if is_angry:
            cv2.putText(annotated_frame, ANGRY_TEXT, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
           
            current_time = time.time()
            if (current_time - last_sound_play_time) > SOUND_COOLDOWN_SECONDS:
                print(f">>> 偵測到接觸，且距離上次播放已超過 {SOUND_COOLDOWN_SECONDS} 秒。播放音效！")
                playsound(AUDIO_FILE, block=False)
                last_sound_play_time = current_time

        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"影像編碼失敗: {e}")
            continue

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
