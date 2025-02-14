import mysql.connector
from mysql.connector import Error


def connect_to_mysql():
    connection = None  # Khởi tạo biến connection
    try:
        # Tạo kết nối đến MySQL
        connection = mysql.connector.connect(
            host='localhost',
            database='QLsinhvien',
            user='root',
            password='',
            connection_timeout=60  # Tăng thời gian chờ kết nối lên 60 giây
        )

        if connection.is_connected():
            print("Kết nối thành công!")

            # Thực thi câu lệnh SQL
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print("Bạn đang kết nối đến CSDL: ", record)

    except Error as e:
        print("Lỗi khi kết nối MySQL:", e)

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("Đã đóng kết nối MySQL.")


# Gọi hàm để kết nối
connect_to_mysql()
