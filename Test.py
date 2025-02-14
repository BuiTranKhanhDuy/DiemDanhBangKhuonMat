import cv2  # Nhập thư viện OpenCV để sử dụng các hàm nhận diện và xử lý hình ảnh.
import pyodbc  # Nhập thư viện pyodbc để kết nối cơ sở dữ liệu SQL Server
import datetime  # Nhập thư viện datetime để làm việc với thời gian

# Đọc mô hình đã huấn luyện
model = cv2.face.LBPHFaceRecognizer_create()
model.read('face_recognition_model.xml')  # Tải mô hình đã huấn luyện từ tệp XML.

# Đọc ánh xạ ID, tên và MSSV từ tệp
id_to_name_mssv = {}
with open('id_name_mssv.txt', 'r', encoding='utf-8') as file:  # Mở tệp chứa ánh xạ ID, tên và MSSV.
    for line in file:  # Duyệt qua từng dòng trong tệp.
        id, name, mssv = line.strip().split(',')  # Tách ID, tên và MSSV từ dòng, loại bỏ khoảng trắng.
        id_to_name_mssv[int(id)] = (name, mssv)  # Lưu vào từ điển với ID là khóa và (tên, MSSV) là giá trị.

# Kết nối đến cơ sở dữ liệu
server = 'LAPTOP-6G3NTB4P\SQLEXPRESS'  # Địa chỉ SQL Server
database = 'QLsinhvien'  # Tên cơ sở dữ liệu
username = 'NguyenChanHung_2151067612'  # Tên người dùng SQL Server
password = '1'  # Mật khẩu của bạn

# Cấu hình chuỗi kết nối
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Khởi tạo webcam
camera = cv2.VideoCapture(0)  # Tạo đối tượng VideoCapture để sử dụng webcam.
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # Tải cascade classifier để phát hiện khuôn mặt.

# Thiết lập ngưỡng độ tin cậy
confidence_threshold = 55  # Thiết lập ngưỡng độ tin cậy.

# Biến để kiểm tra thời gian đứng yên
last_seen_time = None  # Thời gian lần cuối cùng thấy khuôn mặt
standing_still_duration = 5  # Thời gian cần đứng yên (giây)

try:
    # Tạo kết nối
    connection = pyodbc.connect(connection_string)
    print("Kết nối cơ sở dữ liệu thành công!")

    while True:  # Vòng lặp chính để xử lý video từ webcam.
        ret, img = camera.read()  # Đọc khung hình từ webcam.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển khung hình sang đen trắng.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(30, 30))  # Phát hiện các khuôn mặt trong khung hình.

        if len(faces) > 0:  # Nếu phát hiện khuôn mặt
            (x, y, w, h) = faces[0]  # Lấy khuôn mặt đầu tiên
            roi_gray = gray[y:y + h, x:x + w]  # Lấy vùng khuôn mặt từ ảnh đen trắng.

            # Dự đoán ID và độ tin cậy của khuôn mặt
            label, confidence = model.predict(roi_gray)  # Dự đoán ID và độ tin cậy.

            # Nếu độ tin cậy lớn hơn ngưỡng, hiển thị "Unknown"
            if confidence > confidence_threshold:  # Kiểm tra nếu độ tin cậy lớn hơn ngưỡng.
                label_text = "Unknown"  # Nếu lớn hơn ngưỡng, gán tên là "Unknown".
                mssv = None  # Không có MSSV cho "Unknown"
            else:
                # Lấy tên và MSSV tương ứng với ID
                label_text, mssv = id_to_name_mssv.get(label, ("Unknown", None))  # Lấy tên và MSSV tương ứng với ID.

                # Kiểm tra thời gian đứng yên
                current_time = datetime.datetime.now()
                if last_seen_time is None or (current_time - last_seen_time).total_seconds() > standing_still_duration:
                    # Nếu lần đầu tiên thấy hoặc đã đứng yên đủ lâu
                    last_seen_time = current_time  # Cập nhật thời gian thấy

                    # Kiểm tra xem MSSV đã tồn tại trong bảng DiemDanh chưa
                    cursor = connection.cursor()
                    check_query = "SELECT COUNT(*) FROM DiemDanh WHERE MSSV = ? AND CONVERT(date, DayTime) = CONVERT(date, ?)"
                    cursor.execute(check_query, (mssv, current_time))
                    count = cursor.fetchone()[0]

                    if count == 0:  # Nếu MSSV chưa tồn tại trong bảng
                        # Lưu thông tin vào cơ sở dữ liệu
                        insert_query = '''
                        INSERT INTO DiemDanh (MSSV, Tensv, DiemDanh, DayTime)
                        VALUES (?, ?, ?, ?)
                        '''
                        daytime = datetime.datetime.now()  # Lấy thời gian hiện tại
                        cursor.execute(insert_query, (mssv, label_text, 1, daytime))  # Giả sử điểm danh thành công
                        connection.commit()  # Cam kết thay đổi
                    else:
                        print(f"MSSV {mssv} đã tồn tại trong bảng DiemDanh. Không thực hiện điểm danh.")

            # Hiển thị độ tin cậy của kết quả nhận diện
            display_confidence = f"Confidence: {confidence:.2f}"  # Định dạng hiển thị độ tin cậy.

            # Chọn màu dựa trên việc nhận diện thành công hay không
            color = (255, 0, 0) if label_text == "Unknown" else (
            0, 255, 0)  # Màu đỏ cho "Unknown", xanh lá cho nhận diện thành công.

            # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên, độ tin cậy
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Vẽ hình chữ nhật quanh khuôn mặt.
            cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,
                        2)  # Hiển thị tên khuôn mặt.
            cv2.putText(img, display_confidence, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                        1)  # Hiển thị độ tin cậy.

        else:
            last_seen_time = None  # Reset nếu không có khuôn mặt nào

        # Hiển thị kết quả nhận diện trên màn hình
        cv2.imshow('Face Recognition', img)  # Hiển thị khung hình đã xử lý.

        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Kiểm tra phím 'q'.
            break  # Thoát khỏi vòng lặp.

except Exception as e:
    print(f"Lỗi kết nối hoặc thực thi câu lệnh SQL: {e}")

# Giải phóng tài nguyên
camera.release()  # Giải phóng webcam.
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV.

if 'connection' in locals():
    connection.close()
    print("Đã đóng kết nối cơ sở dữ liệu.")
