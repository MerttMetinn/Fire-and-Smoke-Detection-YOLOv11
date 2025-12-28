from ultralytics import YOLO
import cv2
import os

# Modeli yükle
model = YOLO("models/best.pt")

# Kaynak seçimi (0: Webcam, veya 'resim.jpg' / 'video.mp4')
source = 0 

# Tahmin yap
results = model.predict(source=source, show=True, conf=0.55)

# Eğer kaynak bir resim dosyasıysa, pencereyi açık tut
if isinstance(source, str) and os.path.isfile(source):
    print("Kapatmak için herhangi bir tuşa basın...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
