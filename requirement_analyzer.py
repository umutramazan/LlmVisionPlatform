import os
import json
import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv

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

class HardwareConstraints(BaseModel):
    device_name: Optional[str] = Field(
        None, description="Kullanıcının elindeki cihaz. Örn: 'Raspberry Pi 5', 'Jetson Orin Nano'"
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
    hardware: HardwareConstraints = Field(
        default_factory=HardwareConstraints,
        description="Donanım kısıtlamaları ve tercihler."
    )
    suggested_model: Optional[str] = Field(
        None, description="LLM tarafından önerilen model. Örn: 'YOLOv8-Nano'"
    )

# ==========================================
# BÖLÜM 2: OPENAI AJAN MANTIĞI (DÜZELTME)
# ==========================================

class RecipeAgent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.history = []
        self.collected_info = {}  # Toplanan bilgileri saklayalım
        
        # Pydantic şemasını LLM'in anlayacağı JSON formatına çeviriyoruz
        schema_json = VisionProjectRecipe.model_json_schema()
        
        self.system_prompt = f"""
Sen bir Görüntü İşleme Proje Danışmanısın. Kullanıcı teknik bilgiye sahip OLMAYABILIR.

🎯 GÖREV:
Kullanıcıyla doğal bir sohbet yaparak aşağıdaki bilgileri topla ve bir JSON reçetesi oluştur:

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
   (Sen makul değerler belirle: hızlı→30fps/100ms, normal→15fps/300ms, çok hızlı→60fps/50ms)

4. **Donanım ve Deployment**
   - Nerede çalışacak? (küçük cihaz, bilgisayar, sunucu, belirtmemiş)
   - Hangi cihaz varsa? (Raspberry Pi, Jetson, PC, vs.)
   → Buradan çıkar: deployment (edge_device/cloud_api/hybrid), device_name

5. **Model Önerisi**
   - Yukarıdaki bilgilere göre en uygun Computer Vision modelini SEN seç
   → Bildiğin modeller: YOLOv8/v10 (nano/small/medium), MobileNetV3, EfficientNet, 
     PatchCore, EfficientAD, PaddleOCR, EasyOCR, Facenet, ResNet, vb.

🧠 NASIL DAVRANMALISIN:

✅ **YAP:**
- Kullanıcının dilini kullan (teknik/günlük ne söylüyorsa)
- İlk mesajdan maksimum çıkarım yap
- Eksik bilgiler için NET ve KISA sorular sor (1-2 soru)
- Belirsizliklerde akıllıca varsayımlar yap
- Tüm bilgiler toplandığında "[REÇETE HAZIR]" yaz

❌ **YAPMA:**
- Gereksiz teknik jargon kullanma (kullanıcı teknik değilse)
- Zaten söylenen şeyleri tekrar sorma
- Çok fazla soru sorma (kullanıcıyı yorma)
- Kesin bilmediğin şeylerde katı kurallar uygulama

💡 **AKILLI ÇIKARIMLAR:**
- "hatalı ürün bulmak" → anomaly_detection muhtemelen
- "araba saymak" → object_detection kesin
- "plaka okumak" → ocr kesin
- "fabrika içi" → büyük ihtimalle indoor_controlled
- "hızlı" → muhtemelen 30fps civarı
- "Raspberry Pi" → kesinlikle edge_device, küçük model gerek

🎨 SEN KARAR VER:
Kullanıcı her detayı vermeyebilir. Mantıklı olanı SEN seç:
- Proje adını SEN oluştur (task_amac_v1 formatında)
- FPS ve latency değerlerini SEN belirle
- En uygun modeli SEN seç
- Eğer cihaz belirtmediyse, deployment tipini kullanım senaryosuna göre SEN öner

JSON ŞEMASI:
{json.dumps(schema_json, indent=2)}

🔑 ÖNEMLİ:
- Sohbet sırasında JSON döndürme!
- Tüm bilgiler tamamlanınca "[REÇETE HAZIR]" yaz.
- Sonraki adımda JSON oluşturulacak.
"""
        
        self.history.append({"role": "system", "content": self.system_prompt})

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
        self.history.append({"role": "user", "content": user_input})

        try:
            # ✅ response_format KULLANMIYORUZ - LLM'in doğal sohbet etmesine izin veriyoruz
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.history,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": ai_response})
            
            # "[REÇETE HAZIR]" kontrolü
            if "[REÇETE HAZIR]" in ai_response or "[RECETE HAZIR]" in ai_response:
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
            return {"status": "error", "message": f"API Hatası: {str(e)}"}

    def generate_recipe(self):
        """Reçete hazır olduğunda bu fonksiyonu çağır, JSON oluştur"""
        try:
            # JSON üretimi için özel istek
            json_request = {
                "role": "user",
                "content": "Şimdi topladığın tüm bilgileri kullanarak VisionProjectRecipe JSON şemasına uygun bir JSON oluştur. SADECE JSON döndür, başka açıklama yapma."
            }
            
            self.history.append(json_request)
            
            # ✅ Şimdi response_format kullanabiliriz çünkü sadece JSON istiyoruz
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.history,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            json_response = response.choices[0].message.content
            cleaned_json = self._clean_json_string(json_response)
            
            # JSON'u parse et ve validate et
            data = json.loads(cleaned_json)
            recipe = VisionProjectRecipe(**data)
            
            return {
                "status": "completed",
                "message": "✅ Reçete başarıyla oluşturuldu ve doğrulandı!",
                "recipe": recipe
            }
            
        except (json.JSONDecodeError, ValidationError) as e:
            return {
                "status": "error",
                "message": f"❌ JSON oluşturma hatası: {str(e)}\nLütfen daha fazla detay verin.",
                "recipe": None
            }
        except Exception as e:
            return {"status": "error", "message": f"API Hatası: {str(e)}"}

# ==========================================
# BÖLÜM 3: ÇALIŞTIRMA (MAIN LOOP)
# ==========================================

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")

    if not API_KEY or API_KEY == "sk-your-api-key-here":
        print("❌ Lütfen .env dosyasındaki OPENAI_API_KEY değişkenine geçerli bir OpenAI anahtarı girin.")
    else:
        agent = RecipeAgent(API_KEY)
        
        print("\n🤖 GÖRÜNTÜ İŞLEME MİMARI: Merhaba! Projenizden bahsedin, teknik detayları belirleyelim.\n")
        print("Çıkmak için 'q' tuşuna basabilirsiniz.\n")
        
        while True:
            try:
                user_in = input("Siz: ")
            except (KeyboardInterrupt, EOFError):
                print("\nGörüşürüz!")
                break

            if user_in.lower() in ["q", "exit", "çık"]:
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
                    if recipe.hardware.device_name:
                        print(f"💻 Cihaz: {recipe.hardware.device_name}")
                    print(f"🧠 Önerilen Model: {recipe.suggested_model}")
                    print("="*60)
                    
                    # JSON'u kaydet
                    output_file = f"{recipe.project_name}_recipe.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(recipe.model_dump(), f, indent=2, ensure_ascii=False)
                    print(f"\n💾 Reçete kaydedildi: {output_file}")
                    
                    break
                else:
                    print(f"❌ {json_result['message']}")
            
            elif result["status"] == "error":
                print(f"❌ Hata: {result['message']}")
                break