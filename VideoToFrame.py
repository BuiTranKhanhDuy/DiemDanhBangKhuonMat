import cv2
import os

# Đường dẫn video
video_path = 'F:/CDCNTT/4.mp4'  # Thay đổi đường dẫn video nếu cần

# Lấy ID lớn nhất từ file nếu nó đã tồn tại
face_id = 0  # Mặc định ID = 0
id_name_mssv_file = 'id_name_mssv.txt'

if os.path.exists(id_name_mssv_file):
    with open(id_name_mssv_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if lines:
            last_entry = lines[-1]
            last_id = int(last_entry.split(',')[0])  # Lấy ID cuối cùng
            face_id = last_id + 1  # Tăng ID lên 1

# Nhập tên cho khuôn mặt
name = input("Vui lòng nhập tên của bạn: ").strip()

# Nhập MSSV từ người dùng
mssv = input("Vui lòng nhập MSSV của bạn: ").strip()

# Kiểm tra tên đã tồn tại hay chưa
with open(id_name_mssv_file, 'r', encoding='utf-8') as file:
    if any(f"{face_id},{name}," in line for line in file.readlines()):
        print("Tên đã tồn tại.")
        exit()

# Tạo thư mục để lưu ảnh
data_directory = 'data'
subdirectory = os.path.join(data_directory, f"face_{face_id}")
os.makedirs(subdirectory, exist_ok=True)

# Tải Haar Cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Mở video
cap = cv2.VideoCapture(video_path)

# Kiểm tra xem video có mở được không
if not cap.isOpened():
    print("Không thể mở video.")
    exit()

frame_count = 0
face_count = 0
min_face_size = 50  # Ngưỡng tối thiểu cho kích thước khuôn mặt

# Phát hiện và lưu khuôn mặt từ video
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:  # Nếu video kết thúc
        break
    frame_count += 1

    # Phát hiện khuôn mặt trong khung hình màu
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Kiểm tra kích thước khuôn mặt trước khi lưu
        if w < min_face_size or h < min_face_size:
            continue  # Bỏ qua những khuôn mặt quá nhỏ

        # Cắt khuôn mặt từ khung hình màu
        face = frame[y:y + h, x:x + w]  # Giữ nguyên ảnh màu
        face_count += 1

        # Lưu khuôn mặt vào thư mục dưới dạng ảnh màu
        face_filename = os.path.join(subdirectory, f"face_{face_count}.jpg")
        cv2.imwrite(face_filename, face)  # Lưu ảnh màu
        print(f"Lưu khuôn mặt {face_count}: {face_filename}")

# Giải phóng tài nguyên video
cap.release()
cv2.destroyAllWindows()

# Ghi thông tin vào file id_name_mssv.txt
with open(id_name_mssv_file, 'a', encoding='utf-8') as file:
    file.write(f"{face_id},{name},{mssv}\n")  # Ghi ID mới, tên và MSSV vào file

print(f"Đã lưu {face_count} khuôn mặt từ video vào thư mục {subdirectory}.")
print(f"Thông tin đã được lưu vào {id_name_mssv_file}: ID: {face_id}, Name: {name}, MSSV: {mssv}")
