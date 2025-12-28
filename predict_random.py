from ultralytics import YOLO
import cv2
import glob
import random
import os
from datetime import datetime

# Modeli yükle
model = YOLO("models/best.pt")

# Klasör yolları
test_folder = "test_images"
output_folder = "predictions"

# Output klasörünü oluştur (yoksa)
os.makedirs(output_folder, exist_ok=True)

# Test klasöründen tüm resimleri bul (jpg, png, jpeg)
test_images = glob.glob(f"{test_folder}/*.jpg") + \
              glob.glob(f"{test_folder}/*.png") + \
              glob.glob(f"{test_folder}/*.jpeg")

if not test_images:
    print("❌ Test klasöründe resim bulunamadı!")
    print(f"   Lütfen '{test_folder}' klasörüne resim ekleyin.")
    exit()

# Rastgele bir resim seç
random_image = random.choice(test_images)
image_name = os.path.basename(random_image)

print(f"🎲 Seçilen resim: {image_name}")
print(f"📊 Tahmin yapılıyor...")

# Tahmin yap
results = model.predict(
    source=random_image,
    conf=0.55,  # %55 güven eşiği (F1 eğrisinden elde edilen ideal değer)
    save=False  # Kendimiz kaydedeceğiz
)

# Sonucu al ve kaydet
for result in results:
    # Tahminlerin çizildiği resmi al
    annotated_image = result.plot()
    
    # Benzersiz dosya adı oluştur (tarih-saat ile)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"pred_{timestamp}_{image_name}"
    output_path = os.path.join(output_folder, output_filename)
    
    # Resmi kaydet
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ Sonuç kaydedildi: {output_path}")
    
    # Tespit edilen nesneleri listele
    if len(result.boxes) > 0:
        print(f"\n🔍 Tespit edilen nesneler ({len(result.boxes)} adet):")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            print(f"   • {class_name}: %{confidence*100:.1f}")
    else:
        print("\n⚠️ Bu resimde yangın veya duman tespit edilmedi.")
    
    # Resmi göster
    cv2.imshow("Fire and Smoke Detection - Prediction", annotated_image)
    print("\n📌 Kapatmak için herhangi bir tuşa basın...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

