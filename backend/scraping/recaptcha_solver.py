# backend/scraping/recaptcha_solver.py
import os
import random
import time
import requests
from typing import Optional

# 3rd party
import pydub
import speech_recognition as sr
from playwright.sync_api import Page, FrameLocator, sync_playwright

# --- ตั้งค่า FFmpeg ---
# (เหมือนเดิม: เช็คไฟล์ใน folder ปัจจุบันก่อน ถ้าไม่มีให้ใช้จาก system path)
if os.name == 'nt':
    if os.path.exists("ffmpeg.exe"):
        pydub.AudioSegment.converter = os.path.abspath("ffmpeg.exe")
        pydub.AudioSegment.ffprobe = os.path.abspath("ffprobe.exe")
    else:
        pydub.AudioSegment.converter = "ffmpeg"
        pydub.AudioSegment.ffprobe = "ffprobe"

class RecaptchaSolver:
    """
    Advanced Audio ReCaptcha Solver using Playwright
    Features: Request-based download, Human-like typing, Retry logic
    """

    TEMP_DIR = os.getenv("TEMP") if os.name == "nt" else "/tmp"

    def __init__(self, page: Page) -> None:
        """
        :param page: Playwright Page object
        """
        self.page = page
        self.recognizer = sr.Recognizer()

    def solveCaptcha(self, max_retries: int = 3) -> bool:
        """
        Main method to solve captcha with retries.
        Returns: True if solved, False otherwise.
        """
        print("[Solver] Looking for ReCAPTCHA...")

        # 1. เช็คว่ามี Captcha ไหม (รอสูงสุด 3 วินาที)
        try:
            # ReCAPTCHA มักจะอยู่ใน Iframe ที่มี title="reCAPTCHA"
            self.page.wait_for_selector('iframe[title="reCAPTCHA"]', state='attached', timeout=3000)
        except:
            print("[Solver] No ReCAPTCHA found.")
            return True # ถือว่าไม่มี Captcha ให้ผ่านไป

        # ตัวจัดการ Iframe หลัก (Checkbox)
        frame_main = self.page.frame_locator('iframe[title="reCAPTCHA"]')

        # 2. คลิก Checkbox
        try:
            # หา Element กล่อง Checkbox
            checkbox = frame_main.locator(".recaptcha-checkbox-border")
            
            # ขยับเมาส์ไปที่กล่องก่อนคลิก (เนียน)
            if checkbox.is_visible():
                checkbox.hover()
                time.sleep(random.uniform(0.3, 0.7))
                checkbox.click()
                time.sleep(1)
            else:
                print("[Solver] Checkbox not visible, skipping click.")
        except Exception as e:
            print(f"[Solver] Error clicking checkbox: {e}")

        # 3. ตรวจสอบว่าผ่านเลยไหม (ไม่ต้องแก้รูป/เสียง)
        if self.is_solved():
            print("[Solver] Solved by click!")
            return True

        # 4. เริ่มกระบวนการแก้ Audio (วนลูปตามจำนวน max_retries)
        print("[Solver] Click failed. Starting Audio Challenge...")

        # ตัวจัดการ Iframe Challenge (Popup รูป/เสียง)
        # URL มักจะมีคำว่า bframe หรือ title มีคำว่า challenge
        frame_challenge = self.page.frame_locator('iframe[src*="bframe"]')

        # กดปุ่ม Audio
        try:
            btn_audio = frame_challenge.locator("#recaptcha-audio-button")
            btn_audio.wait_for(state="visible", timeout=5000)
            btn_audio.click()
            time.sleep(1.5)
        except Exception:
            print("[Solver] Could not click Audio button (Maybe blocked or network lag).")
            return False

        # วนลูปแก้
        for attempt in range(1, max_retries + 1):
            print(f"[Solver] Audio Attempt {attempt}/{max_retries}")

            # เช็คว่าโดนบล็อคไหม
            if self.is_detected(frame_challenge):
                print("[Solver] ❌ BLOCKED: 'Try again later'. Change IP or wait.")
                return False

            try:
                # หา URL ไฟล์เสียง
                audio_source = frame_challenge.locator("#audio-source")
                audio_source.wait_for(state="attached", timeout=5000)
                src = audio_source.get_attribute("src")

                if not src:
                    raise Exception("No audio source found")

                # ดาวน์โหลดและแปลงเสียง
                text = self._audio_to_text(src)
                print(f"[Solver] Heard: '{text}'")

                if not text:
                    raise Exception("Empty text recognized")

                # พิมพ์ตอบ (แบบคนพิมพ์ ทีละตัวอักษร)
                input_box = frame_challenge.locator("#audio-response")
                input_box.clear()
                
                # จำลองการพิมพ์ทีละตัว
                for char in text.lower():
                    input_box.type(char, delay=random.randint(50, 200)) 
                    # Playwright มี delay ใน type ได้เลย หรือจะใช้ loop sleep เองก็ได้
                
                time.sleep(0.5)

                # กด Verify
                verify_btn = frame_challenge.locator("#recaptcha-verify-button")
                verify_btn.click()
                time.sleep(2)

                # เช็คผลลัพธ์
                if self.is_solved():
                    print("[Solver] ✅ CAPTCHA SOLVED SUCCESSFULLY!")
                    return True

                # ถ้ายังไม่ผ่าน เช็ค Error Message
                err_msg = frame_challenge.locator(".rc-audiochallenge-error-message")
                if err_msg.is_visible() and err_msg.text_content():
                    print(f"[Solver] Google said: {err_msg.text_content()}")

                print("[Solver] Verify failed, clicking Reload for new audio...")
                reload_btn = frame_challenge.locator("#recaptcha-reload-button")
                if reload_btn.is_visible():
                    reload_btn.click()
                    time.sleep(2)

            except Exception as e:
                print(f"[Solver] Attempt failed: {e}")
                # ถ้าเกิด Error ให้กด Reload แล้วลองใหม่
                try:
                    reload_btn = frame_challenge.locator("#recaptcha-reload-button")
                    if reload_btn.is_visible():
                        reload_btn.click()
                        time.sleep(2)
                except:
                    pass

        print("[Solver] ❌ Failed to solve after retries.")
        return False

    def _audio_to_text(self, url: str) -> Optional[str]:
        """Download mp3, convert to wav, recognize text."""
        mp3_path = os.path.join(self.TEMP_DIR, f"audio_{random.randint(1000,9999)}.mp3")
        wav_path = os.path.join(self.TEMP_DIR, f"audio_{random.randint(1000,9999)}.wav")

        try:
            # ใช้ requests เหมือนเดิม ดีแล้ว
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.google.com/"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[Solver] Failed to download audio: status {resp.status_code}")
                return None
            
            with open(mp3_path, "wb") as f:
                f.write(resp.content)

            # Convert
            sound = pydub.AudioSegment.from_mp3(mp3_path)
            sound.export(wav_path, format="wav")

            # Recognize
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                try:
                    text = self.recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    print("[Solver] Google Speech could not understand audio (Noise/Static)")
                    return None
                except sr.RequestError as e:
                    print(f"[Solver] Google Speech API Error: {e}")
                    return None

        except Exception as e:
            print(f"[Solver] Audio processing error: {e}")
            return None
        finally:
            # Clean up
            for p in [mp3_path, wav_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except:
                        pass

    def is_solved(self) -> bool:
        """Check checkmark style in the main frame."""
        try:
            frame_main = self.page.frame_locator('iframe[title="reCAPTCHA"]')
            # เช็คว่ามี class 'recaptcha-checkbox-checked' อยู่ที่ anchor หรือไม่
            # หรือเช็คว่า hidden token มีค่าแล้ว
            
            # วิธีที่ 1: เช็คเครื่องหมายถูก
            checkmark = frame_main.locator(".recaptcha-checkbox-checkmark")
            # ถ้า style เปลี่ยนหรือ class เปลี่ยน
            # ใน Playwright เช็ค attribute ต้องใช้วิธี get_attribute
            is_checked = frame_main.locator('.recaptcha-checkbox-checked').count() > 0
            if is_checked:
                return True
                
            # วิธีที่ 2: เช็ค attribute aria-checked ของ checkbox
            checkbox = frame_main.locator("#recaptcha-anchor")
            if checkbox.get_attribute("aria-checked") == "true":
                return True

            return False
        except:
            return False

    def is_detected(self, frame_challenge: FrameLocator) -> bool:
        """Check for 'Try again later' inside challenge frame."""
        try:
            # ใน Playwright ใช้ locator text=... ได้
            if frame_challenge.locator("text=Try again later").is_visible():
                return True
            if frame_challenge.locator("text=Your computer or network may be sending automated queries").is_visible():
                return True
            return False
        except:
            return False

# --- ตัวอย่างการเรียกใช้งาน (Example Usage) ---
if __name__ == "__main__":
    with sync_playwright() as p:
        # เปิด Browser (ตั้งค่าให้เหมือนคนมากที่สุด)
        browser = p.chromium.launch(headless=False, args=["--start-maximized"])
        
        # สร้าง Context พร้อม User Agent เพื่อความเนียน
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport=None # ให้ขยายตามหน้าต่าง
        )
        
        page = context.new_page()

        # ไปยังเว็บที่มี Captcha (ตัวอย่าง: Google Demo)
        print("Opening Demo Page...")
        page.goto("https://www.google.com/recaptcha/api2/demo")
        
        solver = RecaptchaSolver(page)
        
        # เรียกใช้ Solver
        result = solver.solveCaptcha()
        
        if result:
            print(">>> Main Script: Proceeding to submit form...")
            # ทำงานต่อ... กดปุ่ม Submit ฯลฯ
            # page.click("#submit-btn")
        else:
            print(">>> Main Script: Failed to solve.")

        # รอให้ดูผลลัพธ์สักพัก
        time.sleep(5)
        browser.close()