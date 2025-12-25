"""
Recipe Check Module
-------------------
LLM + heuristics ile reçete inceleme ve düzeltme modülü.

Çıktılar:
- *_corrected_recipe.json  (VisionProjectRecipe şemasına uygun düzeltilmiş reçete)
- *_check_report.json      (is_valid/confidence/issues/changes_made raporu)
"""

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from requirement_analyzer import VisionProjectRecipe


# Logging yapılandırması (modül seviyesinde)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Mevcut handler yoksa ekle
if not logger.handlers:
    file_handler = logging.FileHandler("recipe_check.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# ==========================================
# BÖLÜM 1: VERİ MODELLERİ
# ==========================================

Severity = Literal["info", "warning", "critical"]


class CheckIssue(BaseModel):
    """Tespit edilen bir sorun/uyarı."""
    field: str = Field(..., description="Sorunlu alan adı (örn: 'performance.max_latency_ms')")
    severity: Severity = Field(..., description="Sorunun ciddiyeti: info, warning, critical")
    current_value: Any = Field(None, description="Mevcut değer")
    suggested_value: Any = Field(None, description="Önerilen değer")
    reason: str = Field(..., description="Sorunun açıklaması ve düzeltme gerekçesi")


class CheckReport(BaseModel):
    """Reçete inceleme raporu."""
    is_valid: bool = Field(..., description="Reçete geçerli mi? (kritik sorun yoksa True)")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Güven skoru (0-100)")
    summary: str = Field(..., description="Raporun kısa özeti")
    issues: List[CheckIssue] = Field(default_factory=list, description="Tespit edilen sorunlar listesi")
    changes_made: List[str] = Field(default_factory=list, description="Yapılan değişikliklerin listesi")


class CheckOutput(BaseModel):
    """LLM'den beklenen çıktı formatı."""
    corrected_recipe: Dict[str, Any]
    check_report: CheckReport


# ==========================================
# BÖLÜM 2: RECIPE CHECKER SINIFI
# ==========================================

class RecipeChecker:
    """
    LLM + basit kurallar ile reçete inceleme ve düzeltme.
    
    Kullanım:
        checker = RecipeChecker()
        corrected, report = checker.check_and_correct(recipe)
    """

    def __init__(self, model: Optional[str] = None):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "sk-your-api-key-here":
            logger.error("Geçerli bir OPENAI_API_KEY bulunamadı!")
            raise ValueError("Geçerli bir OPENAI_API_KEY environment variable'ı gerekli.")

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"RecipeChecker başlatıldı. Model: {self.model}")

        # Şemaları JSON formatına çevir
        recipe_schema = VisionProjectRecipe.model_json_schema()
        report_schema = CheckReport.model_json_schema()

        self.system_prompt = f"""
Sen bir Senior Computer Vision Engineer'sın.

🎯 GÖREV:
Kullanıcının VisionProjectRecipe reçetesini mantık süzgecinden geçir, hatalı/eksik/tutarsız alanları düzelt ve detaylı bir rapor oluştur.

📋 KONTROL EDİLECEK KONULAR:

1. **Performans Tutarlılığı**
   - min_fps ve max_latency_ms uyumlu mu?
   - Formül: max_latency_ms ≈ 1000 / min_fps (örn: 30 FPS → 33ms)
   - Kameranın max_camera_fps değeri, hedef min_fps'i karşılıyor mu?

2. **Donanım Uyumluluğu**
   - Edge deployment ise device_name belirtilmiş mi?
   - Seçilen model, donanım kapasitesine uygun mu?
   - GPU gerektiren model için has_gpu: true mu?

3. **Model Seçimi**
   - suggested_model spesifik ve tam sürüm mü? (örn: "YOLOv8n", "YOLOv8s", "EfficientDet-Lite0")
   - Model, görev tipine (task_type) uygun mu?
   - Model, donanım kısıtlamalarına uygun mu?

4. **Kamera Ayarları**
   - Çözünürlük mantıklı mı?
   - Nesne mesafesi ile lens tipi uyumlu mu?
   - Bağlantı tipi belirtilmiş mi?

5. **Genel Tutarlılık**
   - target_objects boş değil mi?
   - environment ve deployment uyumlu mu?
   - Eksik kritik alanlar var mı?

⚠️ KURALLAR:
- ÇIKTIYI SADECE JSON olarak ver, başka açıklama YAPMA
- corrected_recipe kesinlikle VisionProjectRecipe şemasına uysun
- check_report kesinlikle CheckReport şemasına uysun
- Sadece mantıklı değişiklik yap; gereksiz değişiklik YAPMA
- Her değişikliği changes_made listesine ekle
- Kritik sorun yoksa is_valid: true olsun

📊 CİDDİYET SEVİYELERİ:
- critical: Sistem çalışmaz veya ciddi performans sorunu (MUTLAKA düzelt)
- warning: İyileştirme önerisi, sistem çalışır ama optimal değil
- info: Bilgilendirme, küçük öneri

VisionProjectRecipe JSON şeması:
{json.dumps(recipe_schema, ensure_ascii=False, indent=2)}

CheckReport JSON şeması:
{json.dumps(report_schema, ensure_ascii=False, indent=2)}
"""

    @staticmethod
    def _recommended_latency_ms(min_fps: int) -> int:
        """FPS değerine göre önerilen gecikme süresini hesapla."""
        if min_fps <= 0:
            return 1000
        return max(1, int(round(1000.0 / float(min_fps))))

    def _heuristic_checks(self, recipe: VisionProjectRecipe) -> List[CheckIssue]:
        """Basit kurallarla ön kontrol yap."""
        issues: List[CheckIssue] = []

        # 1. FPS / Latency tutarlılığı
        recommended = self._recommended_latency_ms(recipe.performance.min_fps)
        if recipe.performance.max_latency_ms > recommended * 1.5:  # %50 tolerans
            issues.append(
                CheckIssue(
                    field="performance.max_latency_ms",
                    severity="critical",
                    current_value=recipe.performance.max_latency_ms,
                    suggested_value=recommended,
                    reason=f"{recipe.performance.min_fps} FPS hedefi için gecikme çok yüksek. Önerilen: ~{recommended}ms"
                )
            )

        # 2. Kamera FPS kontrolü
        if recipe.camera.max_camera_fps < recipe.performance.min_fps:
            issues.append(
                CheckIssue(
                    field="camera.max_camera_fps",
                    severity="critical",
                    current_value=recipe.camera.max_camera_fps,
                    suggested_value=recipe.performance.min_fps,
                    reason="Kameranın desteklediği FPS, hedef FPS'ten düşük!"
                )
            )

        # 3. Edge deployment için device_name kontrolü
        if recipe.deployment.value == "edge_device" and not recipe.hardware.device_name:
            issues.append(
                CheckIssue(
                    field="hardware.device_name",
                    severity="warning",
                    current_value=None,
                    suggested_value="Jetson Orin Nano 8GB / Raspberry Pi 5 4GB",
                    reason="Edge cihazı belirlenmemiş; model seçimi etkilenebilir."
                )
            )

        # 4. GPU gerektiren model için GPU kontrolü
        gpu_models = ["yolo", "efficientdet", "resnet", "mobilenet"]
        if recipe.suggested_model:
            model_lower = recipe.suggested_model.lower()
            needs_gpu = any(m in model_lower for m in gpu_models)
            if needs_gpu and recipe.hardware.has_gpu is False:
                issues.append(
                    CheckIssue(
                        field="hardware.has_gpu",
                        severity="warning",
                        current_value=False,
                        suggested_value=True,
                        reason=f"{recipe.suggested_model} GPU ile daha iyi performans verir."
                    )
                )

        # 5. Hedef nesne kontrolü
        if not recipe.target_objects:
            issues.append(
                CheckIssue(
                    field="target_objects",
                    severity="critical",
                    current_value=[],
                    suggested_value=["hedef_nesne"],
                    reason="Hedef nesne listesi boş olamaz!"
                )
            )

        # 6. Bağlantı tipi eksikliği
        if not recipe.camera.connection_type:
            issues.append(
                CheckIssue(
                    field="camera.connection_type",
                    severity="info",
                    current_value=None,
                    suggested_value="USB",
                    reason="Kamera bağlantı tipi belirtilmemiş."
                )
            )

        return issues

    def check_and_correct(
        self,
        recipe_input: Union[VisionProjectRecipe, Dict[str, Any], str],
    ) -> Tuple[VisionProjectRecipe, CheckReport]:
        """
        Reçeteyi inceleyip düzeltilmiş reçete + rapor döndürür.

        Args:
            recipe_input: VisionProjectRecipe, dict veya JSON string

        Returns:
            Tuple[VisionProjectRecipe, CheckReport]: Düzeltilmiş reçete ve rapor
        """
        logger.info("Reçete inceleme başlatıldı...")

        # Input'u normalize et
        if isinstance(recipe_input, VisionProjectRecipe):
            recipe = recipe_input
            original_dict = recipe.model_dump()
        elif isinstance(recipe_input, str):
            original_dict = json.loads(recipe_input)
            recipe = VisionProjectRecipe(**original_dict)
        else:
            original_dict = recipe_input
            recipe = VisionProjectRecipe(**original_dict)

        # Heuristic kontroller
        heuristic_issues = self._heuristic_checks(recipe)
        logger.info(f"Heuristic kontrol tamamlandı. {len(heuristic_issues)} sorun bulundu.")

        # LLM'e gönderilecek prompt
        user_prompt = {
            "recipe": original_dict,
            "heuristic_findings": [i.model_dump() for i in heuristic_issues],
            "instructions": {
                "task": "Reçeteyi incele, hataları düzelt, rapor oluştur",
                "output_format": {
                    "corrected_recipe": "VisionProjectRecipe şemasına uygun JSON",
                    "check_report": "CheckReport şemasına uygun JSON"
                },
                "rules": [
                    "Heuristic bulgularını değerlendir ve mantıklıysa uygula",
                    "Performans parametrelerini tutarlı hale getir",
                    "suggested_model mutlaka spesifik versiyon olsun",
                    "Gereksiz değişiklik yapma",
                    "Her değişikliği changes_made'e ekle"
                ]
            }
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Aşağıdaki reçeteyi incele, düzelt ve raporla.\n"
                    "SADECE JSON döndür, başka açıklama yapma!\n\n"
                    f"{json.dumps(user_prompt, ensure_ascii=False, indent=2)}"
                )
            }
        ]

        try:
            logger.debug("OpenAI API'ye istek gönderiliyor...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            logger.debug(f"API yanıtı alındı: {content[:200]}...")

            # Parse ve validate
            payload = json.loads(content)
            parsed = CheckOutput(**payload)

            # Düzeltilmiş reçeteyi validate et
            corrected_recipe = VisionProjectRecipe(**parsed.corrected_recipe)

            # is_valid'i otomatik güncelle
            has_critical = any(i.severity == "critical" for i in parsed.check_report.issues)
            if not has_critical and len(parsed.check_report.changes_made) > 0:
                parsed.check_report.is_valid = True

            logger.info(f"Reçete inceleme tamamlandı. Geçerli: {parsed.check_report.is_valid}")
            return corrected_recipe, parsed.check_report

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Parse/validate hatası: {str(e)}", exc_info=True)

            # Fallback: Heuristics ile minimal düzeltme
            corrected = recipe.model_dump()
            changes: List[str] = []

            # Latency düzelt
            recommended = self._recommended_latency_ms(recipe.performance.min_fps)
            if corrected["performance"]["max_latency_ms"] > recommended * 1.5:
                corrected["performance"]["max_latency_ms"] = recommended
                changes.append(f"performance.max_latency_ms değeri {recommended} olarak düzeltildi.")

            # Bağlantı tipi ekle
            if not corrected["camera"]["connection_type"]:
                corrected["camera"]["connection_type"] = "USB"
                changes.append("camera.connection_type varsayılan olarak 'USB' atandı.")

            report = CheckReport(
                is_valid=False,
                confidence_score=60.0,
                summary="LLM çıktısı doğrulanamadı; temel kurallarla minimal düzeltme yapıldı.",
                issues=heuristic_issues,
                changes_made=changes
            )

            corrected_model = VisionProjectRecipe(**corrected)
            return corrected_model, report

        except Exception as e:
            logger.error(f"Beklenmeyen hata: {str(e)}", exc_info=True)
            raise


# ==========================================
# BÖLÜM 3: YARDIMCI FONKSİYONLAR
# ==========================================

def load_recipe_from_file(path: str) -> VisionProjectRecipe:
    """JSON dosyasından reçete yükle."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return VisionProjectRecipe(**data)


def save_json(path: str, data: Any) -> None:
    """Veriyi JSON dosyasına kaydet."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Dosya kaydedildi: {path}")


# ==========================================
# BÖLÜM 4: TEST (STANDALONE ÇALIŞTIRMA)
# ==========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanım: python recipe_check.py <recipe.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"\n🔎 Reçete inceleniyor: {input_file}\n")

    try:
        checker = RecipeChecker()
        recipe = load_recipe_from_file(input_file)
        corrected, report = checker.check_and_correct(recipe)

        # Dosyaları kaydet
        base_name = recipe.project_name
        corrected_path = f"{base_name}_corrected_recipe.json"
        report_path = f"{base_name}_check_report.json"

        save_json(corrected_path, corrected.model_dump())
        save_json(report_path, report.model_dump())

        print(f"✅ Düzeltilmiş reçete: {corrected_path}")
        print(f"🧾 İnceleme raporu: {report_path}")

    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        sys.exit(1)
