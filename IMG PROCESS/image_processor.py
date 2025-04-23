import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QFileDialog, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing Application")
        self.setGeometry(100, 100, 1200, 600)

        # Initialize variables
        self.original_image = None
        self.processed_image = None

        # Setup UI
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Button layout (horizontal)
        button_layout = QHBoxLayout()

        # Buttons for operations
        upload_button = QPushButton("Upload Image")
        upload_button.clicked.connect(self.upload_image)
        button_layout.addWidget(upload_button)

        histogram_button = QPushButton("Histogram Equalization")
        histogram_button.clicked.connect(self.histogram_equalization)
        button_layout.addWidget(histogram_button)

        threshold_button = QPushButton("Thresholding")
        threshold_button.clicked.connect(self.thresholding)
        button_layout.addWidget(threshold_button)

        negative_button = QPushButton("Negative Image")
        negative_button.clicked.connect(self.negative_image)
        button_layout.addWidget(negative_button)

        log_transform_button = QPushButton("Log Transform")
        log_transform_button.clicked.connect(self.log_transform)
        button_layout.addWidget(log_transform_button)

        contrast_button = QPushButton("Increase Contrast")
        contrast_button.clicked.connect(self.increase_contrast)
        button_layout.addWidget(contrast_button)

        mean_filter_button = QPushButton("Mean Filter")
        mean_filter_button.clicked.connect(self.mean_filter)
        button_layout.addWidget(mean_filter_button)

        median_filter_button = QPushButton("Median Filter")
        median_filter_button.clicked.connect(self.median_filter)
        button_layout.addWidget(median_filter_button)

        gaussian_filter_button = QPushButton("Gaussian Filter")
        gaussian_filter_button.clicked.connect(self.gaussian_filter)
        button_layout.addWidget(gaussian_filter_button)

        bilateral_filter_button = QPushButton("Bilateral Filter")
        bilateral_filter_button.clicked.connect(self.bilateral_filter)
        button_layout.addWidget(bilateral_filter_button)

        non_local_means_button = QPushButton("NonLocalMeans Filter")
        non_local_means_button.clicked.connect(self.non_local_means_filter)
        button_layout.addWidget(non_local_means_button)

        main_layout.addLayout(button_layout)

        # Image display layout (horizontal split)
        image_display_layout = QHBoxLayout()

        self.original_label = QLabel()
        self.original_label.setText("Original Image")

        self.processed_label = QLabel()
        self.processed_label.setText("Processed Image")

        image_display_layout.addWidget(self.original_label)
        image_display_layout.addWidget(self.processed_label)

        main_layout.addLayout(image_display_layout)

        # Set main widget
        container = QWidget()
        container.setLayout(main_layout)

        self.setCentralWidget(container)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_label)

    def display_image(self, image, label):
        if image is None:
            return

        # Convert image to RGB format for display in QLabel
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width

        qimage = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)

        label.setPixmap(pixmap.scaled(label.width(), label.height()))

    def histogram_equalization(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Chuyển ảnh gốc sang ảnh xám (grayscale)
        # 3. Áp dụng cân bằng histogram để cải thiện độ tương phản
        # 4. Lưu ảnh đã xử lý
        # 5. Hiển thị ảnh đã xử lý
        # 6. Vẽ và hiển thị histogram của ảnh gốc và ảnh sau cân bằng


        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Chuyển ảnh gốc sang ảnh xám (grayscale)
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Áp dụng cân bằng histogram để cải thiện độ tương phản
        equalized_image = cv2.equalizeHist(gray_image)

        # Lưu ảnh đã xử lý
        self.processed_image = equalized_image

        # Hiển thị ảnh đã xử lý
        self.display_image(cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB), self.processed_label)

        # Tạo figure để hiển thị histogram
        plt.figure(figsize=(10, 5))

        # Vẽ histogram của ảnh gốc
        plt.subplot(1, 2, 1)
        plt.hist(gray_image.ravel(), 256, [0, 256])
        plt.title('Histogram ban đầu')

        # Vẽ histogram của ảnh sau cân bằng
        plt.subplot(1, 2, 2)
        plt.hist(equalized_image.ravel(), 256, [0, 256])
        plt.title('Histogram sau cân bằng')

        # Hiển thị histogram
        plt.show()

    def thresholding(self):
        # Workflow:
    # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
    # 2. Chuyển ảnh gốc sang ảnh xám (grayscale)
    # 3. Áp dụng ngưỡng nhị phân để phân tách đối tượng và nền
    # 4. Lưu ảnh đã xử lý
    # 5. Hiển thị ảnh đã xử lý


        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Chuyển ảnh gốc sang ảnh xám (grayscale)
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Áp dụng ngưỡng nhị phân để phân tách đối tượng và nền
        _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Lưu ảnh đã xử lý
        self.processed_image = thresholded_image

        # Hiển thị ảnh đã xử lý
        self.display_image(cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB), self.processed_label)

    def negative_image(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Tạo ảnh âm bản bằng cách đảo ngược giá trị pixel
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Tạo ảnh âm bản bằng cách đảo ngược giá trị pixel
        negative_image = cv2.bitwise_not(self.original_image)

        # Lưu ảnh đã xử lý
        self.processed_image = negative_image

        # Hiển thị ảnh đã xử lý
        self.display_image(negative_image, self.processed_label)

    def log_transform(self):

        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Chuyển ảnh gốc sang ảnh xám (grayscale)
        # 3. Áp dụng biến đổi log để tăng độ sáng ở vùng tối
        # 4. Chuẩn hóa ảnh về dải 8-bit
        # 5. Lưu ảnh đã xử lý
        # 6. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Chuyển ảnh gốc sang ảnh xám (grayscale)
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Tính hằng số c cho biến đổi log
        c = 255 / (np.log(1 + np.max(gray_image)))
        # Áp dụng biến đổi log để tăng độ sáng ở vùng tối
        log_transformed = c * (np.log(1 + gray_image.astype(np.float32)))

        # Chuẩn hóa ảnh về dải 8-bit
        log_transformed = np.uint8(log_transformed)

        # Lưu ảnh đã xử lý
        self.processed_image = log_transformed

        # Hiển thị ảnh đã xử lý
        self.display_image(cv2.cvtColor(log_transformed, cv2.COLOR_GRAY2RGB), self.processed_label)

    def increase_contrast(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Chuyển ảnh sang không gian màu YUV
        # 3. Cân bằng histogram trên kênh Y (độ sáng)
        # 4. Chuyển ảnh trở lại không gian màu BGR
        # 5. Lưu ảnh đã xử lý
        # 6. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Chuyển ảnh sang không gian màu YUV
        yuv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YUV)

        # Cân bằng histogram trên kênh Y (độ sáng)
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])

        # Chuyển ảnh trở lại không gian màu BGR
        contrast_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        # Lưu ảnh đã xử lý
        self.processed_image = contrast_image

        # Hiển thị ảnh đã xử lý
        self.display_image(contrast_image, self.processed_label)

    def mean_filter(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Áp dụng bộ lọc trung bình (mean filter) để làm mịn ảnh
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Áp dụng bộ lọc trung bình (mean filter) để làm mịn ảnh
        mean_filtered_image = cv2.blur(self.original_image, (5, 5))

        # Lưu ảnh đã xử lý
        self.processed_image = mean_filtered_image

        # Hiển thị ảnh đã xử lý
        self.display_image(mean_filtered_image, self.processed_label)

    def median_filter(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Áp dụng bộ lọc trung vị (median filter) để giảm nhiễu
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Áp dụng bộ lọc trung vị (median filter) để giảm nhiễu
        median_filtered_image = cv2.medianBlur(self.original_image, 5)

        # Lưu ảnh đã xử lý
        self.processed_image = median_filtered_image

        # Hiển thị ảnh đã xử lý
        self.display_image(median_filtered_image, self.processed_label)

    def gaussian_filter(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Áp dụng bộ lọc Gaussian để làm mịn ảnh
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Áp dụng bộ lọc Gaussian để làm mịn ảnh
        gaussian_filtered_image = cv2.GaussianBlur(self.original_image, (5, 5), 10)

        # Lưu ảnh đã xử lý
        self.processed_image = gaussian_filtered_image

        # Hiển thị ảnh đã xử lý
        self.display_image(gaussian_filtered_image, self.processed_label)

    def bilateral_filter(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Áp dụng bộ lọc song phương (bilateral filter) để làm mịn mà vẫn giữ cạnh
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Áp dụng bộ lọc song phương (bilateral filter) để làm mịn mà vẫn giữ cạnh
        bilateral_filtered_image = cv2.bilateralFilter(self.original_image, 9, 75, 75)

        # Lưu ảnh đã xử lý
        self.processed_image = bilateral_filtered_image

        # Hiển thị ảnh đã xử lý
        self.display_image(bilateral_filtered_image, self.processed_label)

    def non_local_means_filter(self):
        # Workflow:
        # 1. Kiểm tra xem ảnh gốc có tồn tại hay không
        # 2. Áp dụng bộ lọc Non-Local Means để giảm nhiễu ảnh màu
        # 3. Lưu ảnh đã xử lý
        # 4. Hiển thị ảnh đã xử lý

        # Kiểm tra xem ảnh gốc có tồn tại hay không
        if self.original_image is None:
            return

        # Áp dụng bộ lọc Non-Local Means để giảm nhiễu ảnh màu
        non_local_means_filtered_image = cv2.fastNlMeansDenoisingColored(self.original_image, None, 10, 10, 7, 21)

        # Lưu ảnh đã xử lý
        self.processed_image = non_local_means_filtered_image

        # Hiển thị ảnh đã xử lý
        self.display_image(non_local_means_filtered_image, self.processed_label)


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
