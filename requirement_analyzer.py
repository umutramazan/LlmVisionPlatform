import os
import json
import re
import logging
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('requirement_analyzer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# BÖLÜM 1: VERİ MODELLERİ
# ==========================================

class CVTaskType(str, Enum):
    DETECTION = "object_detection"       # Nesne Tespiti
    CLASSIFICATION = "classification"    # Sınıflandırma
    SEGMENTATION = "segmentation"        # Bölütleme
    OCR = "optical_character_recognition" # Yazı Okuma
    ANOMALY_DETECTION = "anomaly_detection" # Anomali Tespiti

class EnvironmentType(str, Enum):
    INDOOR_CONTROLLED = "indoor_controlled" 
    INDOOR_VARIABLE = "indoor_variable"    
    OUTDOOR_DAY = "outdoor_day"            
    OUTDOOR_NIGHT = "outdoor_night"        
    UNDERWATER = "underwater"              

class DeploymentType(str, Enum):
    EDGE = "edge_device"       # Raspberry Pi, Jetson
    CLOUD = "cloud_api"        # Sunucu
    HYBRID = "hybrid"          # Hibrit

class CameraSpecs(BaseModel):
    resolution_width: int = Field(
        1920, description="Kamera çözünürlüğü genişlik (piksel). Örn: 1920, 1280, 640"
    )
    resolution_height: int = Field(
        1080, description="Kamera çözünürlüğü yükseklik (piksel). Örn: 1080, 720, 480"
    )
    max_camera_fps: int = Field(
        30, description="Kameranın desteklediği maksimum FPS değeri.", ge=1, le=240
    )
    lens_type: Optional[str] = Field(
        None, description="Lens tipi. Örn: 'wide-angle', 'fisheye', 'telephoto', 'standard'"
    )
    is_color: bool = Field(
        True, description="Renkli kamera mı yoksa monokrom mu?"
    )
    connection_type: Optional[str] = Field(
        None, description="Bağlantı türü. Örn: 'USB', 'CSI', 'IP/RTSP', 'MIPI'"
    )
    sensor_type: Optional[str] = Field(
        None, description="Sensör tipi. Örn: 'CMOS', 'CCD'"
    )

class HardwareConstraints(BaseModel):
    device_name: Optional[str] = Field(
        None, description="Kullanıcının elindeki cihaz. Örn: 'Raspberry Pi 5', 'Jetson Orin Nano'"
    )
    ram_gb: Optional[int] = Field(
        None, description="Mevcut RAM miktarı (GB cinsinden). Örn: 4, 8, 16", ge=1
    )
    storage_gb: Optional[int] = Field(
        None, description="Mevcut depolama alanı (GB cinsinden). Örn: 32, 64, 128, 256", ge=1
    )
    has_gpu: Optional[bool] = Field(
        None, description="Cihazda GPU var mı? (CUDA, TensorRT, vb. için önemli)"
    )

class PerformanceMetrics(BaseModel):
    min_fps: int = Field(
        ..., description="Sistemin çalışması gereken minimum kare hızı (FPS).", ge=1, le=120
    )
    max_latency_ms: int = Field(
        ..., description="Kabul edilebilir maksimum gecikme süresi (milisaniye)."
    )

class VisionProjectRecipe(BaseModel):
    project_name: str = Field(..., description="Projenin kısa, teknik adı. Örn: 'traffic_counter_v1'")
    description: str = Field(..., description="Projenin ne yapacağının 1-2 cümlelik özeti.")
    task_type: CVTaskType = Field(..., description="Projenin ana görüntü işleme görevi.")
    target_objects: List[str] = Field(
        ..., 
        description="Tespit edilecek nesnelerin listesi. Örn: ['araba', 'kamyon']",
        min_length=1 
    )
    environment: EnvironmentType = Field(..., description="Kameranın çalışacağı ortam koşulları.")
    deployment: DeploymentType = Field(..., description="Projenin çalışacağı platform (Edge/Cloud).")
    performance: PerformanceMetrics = Field(..., description="Hız ve gecikme gereksinimleri.")
    camera: CameraSpecs = Field(
        default_factory=CameraSpecs,
        description="Kamera özellikleri ve teknik spesifikasyonları."
    )
    hardware: HardwareConstraints = Field(
        default_factory=HardwareConstraints,
        description="Donanım kısıtlamaları ve tercihler."
    )
    suggested_model: Optional[str] = Field(
        None, description="LLM tarafından önerilen model."
    )

# ==========================================
# BÖLÜM 2: OPENAI AJAN MANTIĞI (DÜZELTME) 
# ==========================================

class RecipeAgent:
    MAX_HISTORY_LENGTH = 20  # Maksimum konuşma geçmişi sayısı (system prompt hariç)
    
    def __init__(self):
        # API key'i environment'tan güvenli şekilde oku
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "sk-your-api-key-here":
            logger.error("Geçerli bir OPENAI_API_KEY bulunamadı!")
            raise ValueError("Geçerli bir OPENAI_API_KEY environment variable'ı gerekli.")
        
        self.client = OpenAI(api_key=api_key)
        self.history = []
        logger.info("RecipeAgent başarıyla başlatıldı.")
        
        # Pydantic şemasını LLM'in anlayacağı JSON formatına çeviriyoruz
        schema_json = VisionProjectRecipe.model_json_schema()
        
        self.system_prompt = f"""
Sen bir Senior Computer Vision Engineer'sın. ⚠️ ÖNEMLİ: Kullanıcı görüntü işleme konusunda TEKNİK BİLGİYE SAHİP DEĞİL!

🎯 GÖREV:
Kullanıcının GÜNLÜK DİLLE anlattığı projeden maksimum bilgiyi ÇIKARSANABİLDİĞİNCE ÇOK ÇIKARIM YAP, mümkün olduğunca AZ SORU SOR.

📋 TOPLANMASI GEREKEN BİLGİLER:

1. **Proje Amacı**
   - Ne yapmak istiyor? (tespit, sayma, ayırt etme, okuma, hata bulma, vb.)
   - Hangi nesneler/durumlar üzerinde çalışacak?
   → Buradan çıkar: task_type, target_objects, project_name

2. **Çalışma Ortamı**
   - Nerede kullanılacak? (fabrika, yol, ofis, dışarı, vb.)
   - Işık koşulları nasıl? (sabit, değişken, gece/gündüz)
   → Buradan çıkar: environment

3. **Performans Beklentileri**
   - Hız önemli mi? Gerçek zamanlı olmalı mı?
   - Gecikme tolere edilebilir mi?
   → Buradan çıkar: min_fps, max_latency_ms

4. **Kamera Özellikleri**
   - Hangi kamera kullanılacak? Çözünürlük? (Full HD, HD, düşük çözünürlük)
   - Kameranın FPS değeri ne? (30fps, 60fps standart değerler)
   - Özel lens tipi var mı? (wide-angle, fisheye, normal)
   - Bağlantı tipi? (USB, CSI, IP kamera)
   → Buradan çıkar: resolution_width, resolution_height, max_camera_fps, lens_type, connection_type

5. **Donanım ve Deployment**
   - Nerede çalışacak? (küçük cihaz, bilgisayar, sunucu)
   - Hangi cihaz varsa? (Raspberry Pi, Jetson, PC, vs.)
   - RAM ve depolama ne kadar? (4GB/8GB/16GB RAM, 32GB/64GB depolama)
   - GPU var mı?
   → Buradan çıkar: deployment (edge_device/cloud_api/hybrid), device_name, ram_gb, storage_gb, has_gpu

6. **Model Önerisi**
   - Yukarıdaki bilgilere göre en uygun Computer Vision modelini SEN seç.
   Model önerirken sadece bilinen, yaygın ve 'Deployment Type' ile uyumlu modelleri  öner.


🧠 NASIL DAVRANMALISIN:

✅ **YAP:**
- 🔥 İLK MESAJDAN MAKSİMUM ÇIKARIM YAP! 
- Günlük dil kullan, teknik terimlerden kaçın
- Tüm bilgiler toplandığında "[REÇETE HAZIR]" yaz.

❌ **YAPMA:**
- ❌ Teknik terimler kullanma (FPS, çözünürlük, latency, anomaly detection gibi)
- ❌ Kullanıcının zaten promptunda bahsettiği şeyleri sorma


🎨 SEN KARAR VER:
✅ Kullanıcının anlattığı projeden mantıklı çıkarımlar yap.
✅ Eksik teknik detayları makul değerlerle SEN doldur
✅ Varsayımlarını kullanıcıya günlük dille özet olarak göster.
✅ DONANIM ve MODEL seçiminde NET ve SPESIFIK ol - belirsiz ifadeler kullanma!

📌 REÇETE HAZIR OLMADAN ÖNCE KONTROL ET:
- ✓ Donanım seçimi spesifik mi? 
- ✓ Model seçimi net mi? 


JSON ŞEMASI:
{json.dumps(schema_json, indent=2)}

🔑 ÖNEMLİ:
- Sohbet sırasında JSON döndürme!
- Tüm bilgiler tamamlanınca "[REÇETE HAZIR]" yaz.
- Sonraki adımda JSON oluşturulacak.
"""
        
        self.history.append({"role": "system", "content": self.system_prompt})

    def _truncate_history(self):
        """Konuşma geçmişini belirli bir uzunlukta tutar (system prompt korunur)."""
        if len(self.history) > self.MAX_HISTORY_LENGTH + 1:  # +1 for system prompt
            # System prompt'u koru, eski mesajları sil
            system_prompt = self.history[0]
            self.history = [system_prompt] + self.history[-(self.MAX_HISTORY_LENGTH):]
            logger.info(f"Konuşma geçmişi kırpıldı. Mevcut uzunluk: {len(self.history)}")

    def _clean_json_string(self, json_string):
        """LLM bazen ```json ... ``` şeklinde markdown ekler, bunu temizler."""
        json_string = json_string.strip()
        # Markdown kod bloğu kontrolü
        if json_string.startswith("```json"):
            json_string = json_string[7:]  # ```json kısmını at
        elif json_string.startswith("```"):
            json_string = json_string[3:]  # ``` kısmını at
        if json_string.endswith("```"):
            json_string = json_string[:-3]
        return json_string.strip()

    def chat(self, user_input: str):
        logger.info(f"Kullanıcı girişi alındı: {user_input[:50]}..." if len(user_input) > 50 else f"Kullanıcı girişi alındı: {user_input}")
        
        self.history.append({"role": "user", "content": user_input})
        self._truncate_history()  # Geçmişi kontrol et ve gerekirse kırp

        try:
            # ✅ response_format KULLANMIYORUZ - LLM'in doğal sohbet etmesine izin veriyoruz
            logger.debug(f"OpenAI API'ye istek gönderiliyor. History uzunluğu: {len(self.history)}")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.history,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": ai_response})
            logger.info("OpenAI API yanıtı başarıyla alındı.")
            
            # "[REÇETE HAZIR]" kontrolü
            if "[REÇETE HAZIR]" in ai_response or "[RECETE HAZIR]" in ai_response:
                logger.info("Reçete hazır sinyali alındı.")
                # Kullanıcıya bildir ve JSON iste
                return {
                    "status": "ready_for_json",
                    "message": ai_response,
                    "recipe": None
                }
            
            # Normal sohbet modunda devam et
            return {
                "status": "in_progress",
                "message": ai_response,
                "recipe": None
            }

        except Exception as e:
            logger.error(f"API Hatası: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"API Hatası: {str(e)}"}

    def generate_recipe(self):
        """Reçete hazır olduğunda bu fonksiyonu çağır, JSON oluştur"""
        logger.info("JSON reçetesi oluşturma işlemi başlatıldı.")
        try:
            # JSON üretimi için özel istek
            json_request = {
                "role": "user",
                "content": "Şimdi topladığın tüm bilgileri kullanarak VisionProjectRecipe JSON şemasına uygun bir JSON oluştur. SADECE JSON döndür, başka açıklama yapma."
            }
            
            self.history.append(json_request)
            
            # ✅ Şimdi response_format kullanabiliriz çünkü sadece JSON istiyoruz
            logger.debug("JSON formatında yanıt isteniyor...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.history,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            json_response = response.choices[0].message.content
            cleaned_json = self._clean_json_string(json_response)
            logger.debug(f"Temizlenmiş JSON alındı: {cleaned_json[:100]}...")
            
            # JSON'u parse et ve validate et
            data = json.loads(cleaned_json)
            recipe = VisionProjectRecipe(**data)
            
            logger.info(f"Reçete başarıyla oluşturuldu: {recipe.project_name}")
            return {
                "status": "completed",
                "message": "✅ Reçete başarıyla oluşturuldu ve doğrulandı!",
                "recipe": recipe
            }
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"JSON oluşturma/doğrulama hatası: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"❌ JSON oluşturma hatası: {str(e)}\nLütfen daha fazla detay verin.",
                "recipe": None
            }
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {str(e)}", exc_info=True)
            return {"status": "error", "message": f"API Hatası: {str(e)}"}

# ==========================================
# BÖLÜM 3: ÇALIŞTIRMA (MAIN LOOP)
# ==========================================

if __name__ == "__main__":
    try:
        agent = RecipeAgent()  # API key artık constructor içinde yönetiliyor
        logger.info("Uygulama başlatıldı.")
        
        print("\n🤖 GÖRÜNTÜ İŞLEME MİMARI: Merhaba! Projenizden bahsedin, teknik detayları belirleyelim.\n")
        print("Çıkmak için 'q' tuşuna basabilirsiniz.\n")
        
        while True:
            try:
                user_in = input("Siz: ")
            except (KeyboardInterrupt, EOFError):
                logger.info("Kullanıcı uygulamadan çıktı (Ctrl+C).")
                print("\nGörüşürüz!")
                break

            if user_in.lower() in ["q", "exit", "çık"]:
                logger.info("Kullanıcı uygulamadan çıktı.")
                print("Görüşürüz!")
                break
            
            result = agent.chat(user_in)
            
            if result["status"] == "in_progress":
                print(f"\n🤖 Mimar: {result['message']}\n")
            
            elif result["status"] == "ready_for_json":
                print(f"\n🤖 Mimar: {result['message']}\n")
                print("⚙️  JSON reçetesi oluşturuluyor...\n")
                
                # Reçeteyi oluştur
                json_result = agent.generate_recipe()
                
                if json_result["status"] == "completed":
                    print(f"{json_result['message']}")
                    print("="*60)
                    recipe = json_result["recipe"]
                    
                    # Sonuçları göster
                    print(f"📁 Proje: {recipe.project_name}")
                    print(f"📝 Açıklama: {recipe.description}")
                    print(f"🎯 Görev: {recipe.task_type.name}")
                    print(f"🔍 Hedef Nesneler: {', '.join(recipe.target_objects)}")
                    print(f"🌍 Ortam: {recipe.environment.name}")
                    print(f"🚀 Platform: {recipe.deployment.name}")
                    print(f"⚡ FPS Hedefi: {recipe.performance.min_fps}")
                    print(f"⏱️  Max Gecikme: {recipe.performance.max_latency_ms}ms")
                    print(f"\n📷 KAMERA ÖZELLİKLERİ:")
                    print(f"   Çözünürlük: {recipe.camera.resolution_width}x{recipe.camera.resolution_height}")
                    print(f"   Max FPS: {recipe.camera.max_camera_fps}")
                    if recipe.camera.lens_type:
                        print(f"   Lens: {recipe.camera.lens_type}")
                    print(f"   Tip: {'Renkli' if recipe.camera.is_color else 'Monokrom'}")
                    if recipe.camera.connection_type:
                        print(f"   Bağlantı: {recipe.camera.connection_type}")
                    print(f"\n💻 DONANIM:")
                    if recipe.hardware.device_name:
                        print(f"   Cihaz: {recipe.hardware.device_name}")
                    if recipe.hardware.ram_gb:
                        print(f"   RAM: {recipe.hardware.ram_gb} GB")
                    if recipe.hardware.storage_gb:
                        print(f"   Depolama: {recipe.hardware.storage_gb} GB")
                    if recipe.hardware.has_gpu is not None:
                        print(f"   GPU: {'Var' if recipe.hardware.has_gpu else 'Yok'}")
                    print(f"\n🧠 Önerilen Model: {recipe.suggested_model}")
                    print("="*60)
                    
                    # JSON'u kaydet
                    output_file = f"{recipe.project_name}_recipe.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(recipe.model_dump(), f, indent=2, ensure_ascii=False)
                    print(f"\n💾 Reçete kaydedildi: {output_file}")
                    logger.info(f"Reçete dosyaya kaydedildi: {output_file}")
                    
                    break
                else:
                    print(f"❌ {json_result['message']}")
            
            elif result["status"] == "error":
                print(f"❌ Hata: {result['message']}")
                break
                
    except ValueError as e:
        print(f"❌ {str(e)}")
        print("Lütfen .env dosyasındaki OPENAI_API_KEY değişkenine geçerli bir OpenAI anahtarı girin.")
        exit(1)
