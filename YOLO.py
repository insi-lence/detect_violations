from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Masks
import base64
from flask_cors import CORS 
import sqlite3
from datetime import datetime
from openpyxl import Workbook


app = Flask(__name__)
CORS(app)

# ... Загрузка модели ...
model = YOLO('yolov8n.pt')  

# ... Функция для тернировки модели ...
def train_model():
    model.train(data="./bus_line/data.yaml", epochs=350, imgsz=640, device="mps")


# ... Инициализация базы данных ...

def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            object_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ... Поиск автобусной полосы ...
def detect_bus_lane(image):
    model = YOLO('runs/segment/train5/weights/best.pt')
    results = model.predict(image, iou=0.5, conf=0.85)[0]
    try:
        masks = torch.rand(results[0].masks.shape)
        orig_shape = results[0].masks.orig_shape
        mask_obj = Masks(masks, orig_shape)
        return mask_obj.data[0].numpy()
    except:
        return None


# ... Проверка нахождения мотоцикла на полосе ...
def is_motorcycle_in_bus_lane(mc_box, bus_lane_masks):
    x1, y1, x2, y2 = map(int, mc_box)
    if bus_lane_masks is not None:
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(bus_lane_masks.shape[1], x2); y2 = min(bus_lane_masks.shape[0], y2)
        roi = bus_lane_masks[y1:y2, x1:x2]
        return np.any(roi > 0)


def get_result(image, filename):
    results = model.predict(image, iou=0.5)[0]

    # ... Детекция полосы ...
    bus_lane_boxes = detect_bus_lane(image)

    # ... Детекция мотоциклов ..
    motorcycle_boxes = []
    for result in results:
        if result.boxes.cls.cpu().numpy() == 3: # ищем объект с классом 3: мотоцикл
            mc_box = result.boxes.xyxy[0].numpy().astype(int) # сохраняем координаты мотоцикла
            motorcycle_boxes.append(list(mc_box)) # добавляем данные в массив, тут будет список всех найденных мотоциклов

    # ... Проверка нарушений ...
    violations = []
    for mc_box in motorcycle_boxes:
        if is_motorcycle_in_bus_lane(mc_box, bus_lane_boxes):
            violations.append(mc_box)

    # ... Отрисовка результатов ...
    for box in motorcycle_boxes:
        color = (0, 0, 255) if box in violations else (0, 255, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        if box in violations:
            status = "violations"
            cv2.putText(image, status, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else: 
            status = "OK"
            cv2.putText(image, status, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    #cv2.imwrite('./result', image)

    # ... сохранение историии ...
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (timestamp, filename, object_count)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), filename, status))
    conn.commit()

     # ... сохранение отчета ...   
    cursor.execute('SELECT * FROM history')
    rows = cursor.fetchall()
    conn.close()

    wb = Workbook()
    ws = wb.active
    ws.title = "Учёт нарушений"

    ws.append(['ID', 'Дата и время', 'Имя файла', 'Статус'])

    for row in rows:
        ws.append(row)

    filename_xlsx = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb.save(filename_xlsx)
    print(f"Отчёт сохранён в файл: {filename_xlsx}")

    return image



@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    filename = file.filename.lower()
    # ... обработка YOLO ...
    processed_img = get_result(img, filename)



    # ... обработка результата ...
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'processed_image': img_base64})


if __name__ == '__main__':
            app.run(debug=True,host='0.0.0.0', port=5001)

