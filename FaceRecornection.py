import cv2  # Nhập thư viện OpenCV để sử dụng các hàm nhận diện và xử lý hình ảnh.

# Đọc mô hình đã huấn luyện
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognition_model.xml')  # Tải mô hình đã huấn luyện từ tệp XML.

# Đọc ánh xạ ID và tên từ tệp
id_to_name_mssv = {}
with open('id_name_mssv.txt', 'r', encoding='utf-8') as file:  # Mở tệp chứa ánh xạ ID và tên.
    for line in file:  # Duyệt qua từng dòng trong tệp.
        id, name, mssv = line.strip().split(',')  # Tách ID, tên và MSSV từ dòng, loại bỏ khoảng trắng.
        id_to_name_mssv[int(id)] = (name, mssv)  # Lưu vào từ điển với ID là khóa và (tên, MSSV) là giá trị.

# Khởi tạo webcam
camera = cv2.VideoCapture(0)  # Tạo đối tượng VideoCapture để sử dụng webcam.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Tải cascade classifier để phát hiện khuôn mặt.

# Thiết lập ngưỡng độ tin cậy
confidence_threshold = 55  # Thiết lập ngưỡng độ tin cậy.

while True:  # Vòng lặp chính để xử lý video từ webcam.
    ret, img = camera.read()  # Đọc khung hình từ webcam.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển khung hình sang đen trắng.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Phát hiện các khuôn mặt trong khung hình.

    for (x, y, w, h) in faces:  # Duyệt qua danh sách các khuôn mặt đã phát hiện.
        roi_gray = gray[y:y + h, x:x + w]  # Lấy vùng khuôn mặt từ ảnh đen trắng.

        # Dự đoán ID và độ tin cậy của khuôn mặt
        label, confidence = model.predict(roi_gray)  # Dự đoán ID và độ tin cậy.

        # Nếu độ tin cậy lớn hơn ngưỡng, hiển thị "Unknown"
        if confidence > confidence_threshold:  # Kiểm tra nếu độ tin cậy lớn hơn ngưỡng.
            label_text = "Unknown"  # Nếu lớn hơn ngưỡng, gán tên là "Unknown".
        else:
            label_text, _ = id_to_name_mssv.get(label, ("Unknown", ""))  # Lấy tên tương ứng với ID, bỏ qua MSSV.

        # Hiển thị độ tin cậy của kết quả nhận diện
        display_confidence = f"Confidence: {confidence:.2f}"  # Định dạng hiển thị độ tin cậy.

        # Chọn màu dựa trên việc nhận diện thành công hay không
        color = (255, 0, 0) if label_text == "Unknown" else (0, 255, 0)  # Màu đỏ cho "Unknown", xanh lá cho nhận diện thành công.

        # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên và độ tin cậy
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Vẽ hình chữ nhật quanh khuôn mặt.
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Hiển thị tên khuôn mặt.
        cv2.putText(img, display_confidence, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)  # Hiển thị độ tin cậy.

    # Hiển thị kết quả nhận diện trên màn hình
    cv2.imshow('Face Recognition', img)  # Hiển thị khung hình đã xử lý.

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Kiểm tra phím 'q'.
        break  # Thoát khỏi vòng lặp.

# Giải phóng tài nguyên
camera.release()  # Giải phóng webcam.
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV.
