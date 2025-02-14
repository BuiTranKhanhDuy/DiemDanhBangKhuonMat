import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


# Hàm để thực hiện augment hình ảnh
def augment_image(image):
    augmented_images = []

    # Original image
    augmented_images.append(image)

    # Flip the image
    flipped_image = cv2.flip(image, 1)  # Flip horizontally
    augmented_images.append(flipped_image)

    # Rotate image
    for angle in [-15, 15]:  # Rotate by -15 and +15 degrees
        height, width = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        augmented_images.append(rotated_image)

    # Change brightness
    for brightness in [0.5, 1.5]:  # Reduce and increase brightness
        bright_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        augmented_images.append(bright_image)

    return augmented_images


# Hàm để chuẩn bị dữ liệu huấn luyện (với augmentation)
def prepare_training_data(data_directory):
    faces = []
    labels = []

    if not os.path.exists(data_directory):
        print(f"Thư mục '{data_directory}' không tồn tại.")
        return faces, labels

    for label_dir in os.listdir(data_directory):
        label_dir_path = os.path.join(data_directory, label_dir)
        if os.path.isdir(label_dir_path):
            for image_file in os.listdir(label_dir_path):
                image_path = os.path.join(label_dir_path, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Không thể đọc ảnh: {image_path}")
                        continue

                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Prepare augmented images
                    augmented_faces = augment_image(gray_image)
                    faces.extend(augmented_faces)
                    labels.extend(
                        [int(label_dir.split('_')[1])] * len(augmented_faces))  # Thêm label cho từng ảnh đã augment

    return faces, labels


# Đường dẫn đến file XML và thư mục chứa dữ liệu
model_file_path = 'face_recognition_model.xml'
data_directory = 'data'

# Bước 2: (Tùy chọn) Tạo một mô hình mới và huấn luyện
print("Chuẩn bị dữ liệu huấn luyện...")
faces, labels = prepare_training_data(data_directory)

if len(faces) == 0 or len(labels) == 0:
    print("Không có dữ liệu để huấn luyện.")
else:
    print("Huấn luyện mô hình...")

    # Adjusting LBPH parameters for better accuracy
    model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    # Huấn luyện mô hình
    model.train(faces, np.array(labels))

    # Lưu mô hình vào file
    model.save(model_file_path)

    print("Huấn luyện hoàn tất và mô hình đã được lưu.")
