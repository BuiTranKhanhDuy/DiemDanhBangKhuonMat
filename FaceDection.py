import cv2
import os
import winsound

# Khởi động camera và nạp bộ phát hiện khuôn mặt
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera.set(3, 640)  # Đặt độ phân giải camera
camera.set(4, 480)

# Kiểm tra xem camera có mở được không
if not camera.isOpened():
    print("Không thể mở camera.")
    exit()

# Đường dẫn lưu trữ thông tin người dùng
id_name_mssv_file = 'id_name_mssv.txt'

# Lấy ID lớn nhất từ file nếu nó đã tồn tại
face_id = 0  # Mặc định ID = 0
if os.path.exists(id_name_mssv_file):
    with open(id_name_mssv_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if lines:
            last_entry = lines[-1]
            last_id = int(last_entry.split(',')[0])  # Lấy ID cuối cùng
            face_id = last_id + 1  # Tăng ID lên 1

# Nhập tên và MSSV cho khuôn mặt
name = input("Vui lòng nhập tên của bạn: ").strip()
mssv = input("Vui lòng nhập MSSV của bạn: ").strip()

# Kiểm tra tên đã tồn tại hay chưa
with open(id_name_mssv_file, 'r', encoding='utf-8') as file:
    if any(f"{face_id},{name}," in line for line in file.readlines()):
        print("Tên đã tồn tại.")
        exit()

# Tạo thư mục lưu trữ ảnh
face_directory = 'data'
subdirectory = os.path.join(face_directory, f"face_{face_id}")
os.makedirs(subdirectory, exist_ok=True)

print("Camera đang bật, vui lòng chờ...")
count = 0
max_images = 300  # Số lượng ảnh tối đa

while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh về đen trắng
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]  # Cắt vùng khuôn mặt
        count += 1

        # Tạo và lưu ảnh xám
        img_path = os.path.join(subdirectory, f"image_{count}.jpg")
        cv2.imwrite(img_path, roi_gray)

        # Hiển thị thông tin khuôn mặt
        cv2.putText(img, f"ID: {face_id} - Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('Detected Faces', img)

        # Kiểm tra nếu đã lưu đủ số lượng ảnh
        if count >= max_images:
            break

    # Thoát khi nhấn 'q' hoặc đã chụp đủ ảnh
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= max_images:
        break

# Phát âm thanh khi hoàn tất
winsound.Beep(1000, 500)
print(f"\n Dữ liệu khuôn mặt cho {name} (ID: {face_id}) đã được lưu.")

# Ghi thông tin vào file id_name_mssv.txt
with open(id_name_mssv_file, 'a', encoding='utf-8') as file:
    file.write(f"{face_id},{name},{mssv}\n")  # Ghi ID mới, tên và MSSV vào file

print(f"Thông tin đã được lưu vào {id_name_mssv_file}: ID: {face_id}, Name: {name}, MSSV: {mssv}")

# Giải phóng tài nguyên
camera.release()
cv2.destroyAllWindows()
