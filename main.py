from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO("models/best.pt")

# Kaynak seçimi (0: Webcam, veya 'video.mp4')
source = 0 

# Tahmin yap ve göster
results = model.predict(source=source, show=True, conf=0.55)

# Çıkış için bir tuşa basılmasını bekle (Resimse)
cv2.waitKey(0)
