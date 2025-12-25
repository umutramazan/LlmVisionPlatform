"""
LLM Vision Platform - Main Entry Point
---------------------------------------
Görüntü işleme projesi için reçete oluşturma ve inceleme platformu.

Kullanım:
    python main.py                    # Etkileşimli mod (sohbet ile reçete oluştur)
    python main.py <recipe.json>      # Mevcut reçeteyi incele ve düzelt
"""

import json
import logging
import sys
from typing import Tuple

from requirement_analyzer import RecipeAgent, VisionProjectRecipe
from recipe_check import RecipeChecker, CheckReport, save_json


# Logging yapılandırması (modül seviyesinde)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Mevcut handler yoksa ekle
if not logger.handlers:
    file_handler = logging.FileHandler("llm_vision_platform.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


# ==========================================
# BÖLÜM 1: YARDIMCI FONKSİYONLAR
# ==========================================

def get_output_paths(project_name: str) -> Tuple[str, str, str]:
    """Çıktı dosya yollarını oluştur."""
    recipe_path = f"{project_name}_recipe.json"
    corrected_path = f"{project_name}_corrected_recipe.json"
    report_path = f"{project_name}_check_report.json"
    return recipe_path, corrected_path, report_path


def print_recipe_summary(recipe: VisionProjectRecipe, title: str = "REÇETE ÖZETİ") -> None:
    """Reçete özetini ekrana bas."""
    print("=" * 60)
    print(f"📋 {title}")
    print("=" * 60)
    print(f"📁 Proje: {recipe.project_name}")
    print(f"📝 Açıklama: {recipe.description}")
    print(f"🎯 Görev: {recipe.task_type.value}")
    print(f"🔍 Hedef Nesneler: {', '.join(recipe.target_objects)}")
    print(f"🌍 Ortam: {recipe.environment.value}")
    print(f"🚀 Platform: {recipe.deployment.value}")
    print(f"⚡ Hız hedefi: {recipe.performance.min_fps} FPS")
    print(f"⏱️  Gecikme hedefi: {recipe.performance.max_latency_ms} ms")
    
    print(f"\n📷 KAMERA:")
    print(f"   Sayı: {recipe.camera.num_cameras}")
    if recipe.camera.distance_to_object_meters:
        print(f"   Mesafe: {recipe.camera.distance_to_object_meters}m")
    print(f"   Çözünürlük: {recipe.camera.resolution_width}x{recipe.camera.resolution_height}")
    print(f"   Max FPS: {recipe.camera.max_camera_fps}")
    if recipe.camera.lens_type:
        print(f"   Lens: {recipe.camera.lens_type}")
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
    print("=" * 60)


def print_check_report(report: CheckReport) -> None:
    """İnceleme raporunu ekrana bas."""
    print("\n" + "=" * 60)
    print("🔎 İNCELEME RAPORU")
    print("=" * 60)
    
    status = "✅ GEÇERLİ" if report.is_valid else "⚠️ DÜZELTME GEREKLİ"
    print(f"Durum: {status}")
    print(f"Güven Skoru: {report.confidence_score:.1f}/100")
    print(f"Özet: {report.summary}")
    
    if report.issues:
        print(f"\n📋 Tespit Edilen Sorunlar ({len(report.issues)}):")
        for i, issue in enumerate(report.issues, 1):
            severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
            icon = severity_icon.get(issue.severity, "⚪")
            print(f"   {i}. {icon} [{issue.field}] {issue.reason}")
            if issue.suggested_value is not None:
                print(f"      Mevcut: {issue.current_value} → Önerilen: {issue.suggested_value}")
    
    if report.changes_made:
        print(f"\n✏️  Yapılan Değişiklikler ({len(report.changes_made)}):")
        for change in report.changes_made:
            print(f"   • {change}")
    
    print("=" * 60)


# ==========================================
# BÖLÜM 2: ETKİLEŞİMLİ MOD
# ==========================================

def run_interactive() -> int:
    """Etkileşimli mod: Sohbet ile reçete oluştur ve incele."""
    print("\n" + "=" * 60)
    print("🤖 LLM VİZYON PLATFORMU")
    print("=" * 60)
    
    try:
        agent = RecipeAgent()
        checker = RecipeChecker()
    except ValueError as e:
        print(f"❌ Başlatma hatası: {str(e)}")
        return 1

    print("\n🤖 Mimar: Merhaba! Görüntü işleme projenizden bahsedin, birlikte tasarlayalım.\n")
    print("💡 İpucu: Çıkmak için 'q' yazabilirsiniz.\n")

    while True:
        try:
            user_in = input("Siz: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGörüşürüz! 👋")
            return 0

        if not user_in:
            continue

        if user_in.lower() in ["q", "exit", "quit", "çık", "cik"]:
            print("Görüşürüz! 👋")
            return 0

        # Sohbet devam
        result = agent.chat(user_in)

        if result["status"] == "in_progress":
            print(f"\n🤖 Mimar: {result['message']}\n")
            continue

        if result["status"] == "error":
            print(f"\n❌ Hata: {result['message']}\n")
            continue

        # Reçete hazır sinyali alındı
        if result["status"] == "ready_for_json":
            print(f"\n🤖 Mimar: {result['message']}\n")
            print("⚙️  Reçete oluşturuluyor...\n")

            # JSON reçetesi oluştur
            json_result = agent.generate_recipe()

            if json_result["status"] != "completed":
                print(f"❌ {json_result['message']}")
                continue

            recipe = json_result["recipe"]
            recipe_path, corrected_path, report_path = get_output_paths(recipe.project_name)

            # Ham reçeteyi kaydet
            save_json(recipe_path, recipe.model_dump())
            print(f"💾 Ham reçete kaydedildi: {recipe_path}")

            # Reçete özetini göster
            print_recipe_summary(recipe, "OLUŞTURULAN REÇETE")

            # LLM ile inceleme ve düzeltme
            print("\n🔎 Reçete LLM ile inceleniyor ve düzeltiliyor...\n")
            
            try:
                corrected_recipe, report = checker.check_and_correct(recipe)

                # Düzeltilmiş reçete ve raporu kaydet
                save_json(corrected_path, corrected_recipe.model_dump())
                save_json(report_path, report.model_dump())

                # Raporu göster
                print_check_report(report)

                # Düzeltilmiş reçete özetini göster (değişiklik varsa)
                if report.changes_made:
                    print_recipe_summary(corrected_recipe, "DÜZELTİLMİŞ REÇETE")

                print(f"\n💾 Kaydedilen dosyalar:")
                print(f"   📄 Ham reçete: {recipe_path}")
                print(f"   ✅ Düzeltilmiş: {corrected_path}")
                print(f"   🧾 Rapor: {report_path}\n")

            except Exception as e:
                logger.error(f"İnceleme hatası: {str(e)}", exc_info=True)
                print(f"⚠️  İnceleme sırasında hata: {str(e)}")
                print(f"   Ham reçete yine de kaydedildi: {recipe_path}\n")

            return 0

    return 0


# ==========================================
# BÖLÜM 3: DOSYA MODU
# ==========================================

def run_from_file(file_path: str) -> int:
    """Dosya modu: Mevcut reçeteyi incele ve düzelt."""
    print("\n" + "=" * 60)
    print("🔎 REÇETE İNCELEME MODU")
    print("=" * 60)
    print(f"\n📂 Dosya: {file_path}\n")

    try:
        # Reçeteyi yükle
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        recipe = VisionProjectRecipe(**data)
        print_recipe_summary(recipe, "MEVCUT REÇETE")

        # İnceleme ve düzeltme
        print("\n🔎 LLM ile inceleniyor...\n")
        checker = RecipeChecker()
        corrected_recipe, report = checker.check_and_correct(recipe)

        # Çıktı dosyalarını kaydet
        _, corrected_path, report_path = get_output_paths(recipe.project_name)
        save_json(corrected_path, corrected_recipe.model_dump())
        save_json(report_path, report.model_dump())

        # Raporu göster
        print_check_report(report)

        # Düzeltilmiş reçete özetini göster (değişiklik varsa)
        if report.changes_made:
            print_recipe_summary(corrected_recipe, "DÜZELTİLMİŞ REÇETE")

        print(f"\n💾 Kaydedilen dosyalar:")
        print(f"   ✅ Düzeltilmiş: {corrected_path}")
        print(f"   🧾 Rapor: {report_path}\n")

        return 0

    except FileNotFoundError:
        print(f"❌ Dosya bulunamadı: {file_path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse hatası: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Hata: {str(e)}", exc_info=True)
        print(f"❌ Hata: {str(e)}")
        return 1


# ==========================================
# BÖLÜM 4: MAIN
# ==========================================

def main() -> int:
    """Ana giriş noktası."""
    if len(sys.argv) >= 2:
        # Dosya modu
        return run_from_file(sys.argv[1])
    else:
        # Etkileşimli mod
        return run_interactive()


if __name__ == "__main__":
    sys.exit(main())
