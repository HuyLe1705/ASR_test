import streamlit as st
import os
import tempfile
from faster_whisper import WhisperModel
import torch
import google.generativeai as genai
import time
from audio_recorder_streamlit import audio_recorder
import datetime
import difflib
from pyannote.audio import Pipeline
import torchaudio
import subprocess  # <-- THÊM MỚI

# =========================
# CẤU HÌNH CHUNG
# =========================
st.set_page_config(layout="wide")
st.title("🎤 Voice Transcription (Whisper + Diarization + AI Corrector)")
st.markdown("Nhận dạng giọng nói, phân biệt người nói, và sửa lỗi.")
log_filename = "log.txt"

# =========================
# CẤU HÌNH SIDEBAR
# =========================
with st.sidebar:
    st.header("Cấu hình Whisper")
    
    model_size = st.selectbox("Chọn model Whisper:", 
                              ["tiny", "base", "small", "medium", "large-v3"], 
                              index=2,
                              help="Model lớn hơn (large-v3) chính xác hơn nhưng chậm hơn.")
    
    is_cuda_available = torch.cuda.is_available()
    device_option = st.radio("Thiết bị xử lý (Whisper):", 
                             ["GPU (CUDA)", "CPU"], 
                             index=0 if is_cuda_available else 1,
                             disabled=not is_cuda_available,
                             help="GPU (CUDA) nhanh hơn rất nhiều.")
    
    device = "cuda" if device_option == "GPU (CUDA)" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    st.info(f"Whisper dùng: {device.upper()} ({compute_type})")

    beam_size = st.slider("Beam Size (Whisper):", 
                          min_value=1, 
                          max_value=10, 
                          value=5, 
                          help="Tăng giá trị để tăng độ chính xác.")

    st.divider()
    
    st.header("Cấu hình AI sửa lỗi")
    use_gemini = st.checkbox("Sửa lỗi chính tả bằng AI", value=True)
    
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        gemini_api_key = None 
        
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        hf_token = None
        
    gemini_model_name = "gemini-2.5-flash" 

# =========================
# HÀM TẢI MODEL (CACHE)
# =========================
@st.cache_resource
def load_whisper_model(model_size, device, compute_type):
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as e:
        st.error(f"Lỗi khi tải model Whisper: {e}.")
        return None

# (ĐÃ SỬA LỖI TOKEN)
@st.cache_resource
def load_diarization_model(token):
    if not token:
        st.warning("Thiếu HF_TOKEN trong secrets.toml. Không thể phân biệt người nói.", icon="⚠️")
        return None
    try:
        # Tải pipeline và gửi token (SỬA LẠI THÀNH 'use_auth_token')
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token  # <-- SỬA LẠI CHO PHIÊN BẢN 3.1.1
        )
        
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            st.info("Pyannote: Đã chuyển sang GPU (CUDA).")
        else:
            st.info("Pyannote: Đang dùng CPU.")

        return pipeline
    except Exception as e:
        st.error(f"Lỗi tải model diarization: {e}. Bạn đã đồng ý điều khoản trên Hugging Face chưa?", icon="🔥")
        return None

# Tải cả 2 model
whisper_model = load_whisper_model(model_size, device, compute_type)
diarization_pipeline = load_diarization_model(hf_token)

if whisper_model:
    st.success(f"✅ Model Whisper `{model_size}` đã sẵn sàng.")
if diarization_pipeline:
    st.success("✅ Model Phân biệt người nói (pyannote) đã sẵn sàng.")

# =========================
# HÀM HỖ TRỢ ĐỊNH DẠNG THỜI GIAN
# =========================
def format_timestamp(seconds_float):
    td = datetime.timedelta(seconds=seconds_float)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# =========================
# HÀM NHẬN DẠNG (WHISPER)
# =========================
def run_whisper(model, file_path, beam_size):
    segments, info = model.transcribe(file_path, 
                                      language="vi", 
                                      vad_filter=True, 
                                      beam_size=beam_size,
                                      word_timestamps=True) 
    return list(segments)

# =========================
# HÀM SỬA LỖI (GEMINI)
# =========================
def correct_spelling_with_gemini(text_to_correct, api_key):
    if not text_to_correct:
        return ""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_name)
        
        prompt = f"""Bạn là một chuyên gia sửa lỗi chính tả tiếng Việt.
Nhiệm vụ của bạn là rà soát và sửa các lỗi CHÍNH TẢ (ví dụ: sai 's'/'x', 'tr'/'ch', 'r'/'d'/'gi', dấu hỏi/ngã, v.v.) trong văn bản dưới đây.
QUAN TRỌNG:
1. Chỉ sửa các từ bị sai chính tả.
2. TUYỆT ĐỐI KHÔNG thay đổi cấu trúc câu, không thêm bớt từ.
3. Phải giữ nguyên văn phong và cách diễn đạt gốc của người nói.
4. Có thể thêm các dấu câu (phẩy, chấm, hỏi) nếu nó làm câu rõ nghĩa.
Chỉ trả về văn bản đã được sửa lỗi, không thêm bất kỳ lời giải thích nào.
---
Văn bản gốc:
{text_to_correct}
---
Văn bản đã sửa:
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.toast(f"⚠️ Lỗi Gemini: {e}", icon="🔥")
        return text_to_correct

# =========================
# HÀM XÂY DỰNG TIMELINE ĐÃ SỬA
# =========================
def build_corrected_timeline_html(segment, speaker_label, corrected_segment_text):
    all_original_words = []
    if segment.words:
        all_original_words.extend(segment.words)
    
    original_text_list = [word.word for word in all_original_words]
    corrected_text_list = corrected_segment_text.split() 

    matcher = difflib.SequenceMatcher(None, original_text_list, corrected_text_list, autojunk=False)
    
    seg_start = format_timestamp(segment.start)
    seg_end = format_timestamp(segment.end)
    html = f"<div style='background-color:#222; border-left: 3px solid #00FF00; padding: 10px; border-radius: 5px; font-family: monospace; margin-bottom: 5px;'>"
    html += f"<p style='margin-bottom: 5px;'><strong style='color: cyan;'>[{speaker_label}]</strong> <strong>[{seg_start} -> {seg_end}]</strong>"
    
    try:
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    word = all_original_words[i]
                    start = format_timestamp(word.start)
                    end = format_timestamp(word.end)
                    html += f" <span title='{start} -> {end}' style='cursor: help;'>{word.word}</span>"
            elif tag == 'replace':
                word_start_obj = all_original_words[i1]
                start = format_timestamp(word_start_obj.start)
                word_end_obj = all_original_words[i2-1]
                end = format_timestamp(word_end_obj.end)
                new_words = " ".join(corrected_text_list[j1:j2])
                html += f" <span title='{start} -> {end}' style='cursor: help; color: #00FF00; font-weight: bold;'>{new_words}</span>"
            elif tag == 'insert':
                new_words = " ".join(corrected_text_list[j1:j2])
                html += f" <span style='color: #999999; font-style: italic;'>{new_words}</span>"
            elif tag == 'delete':
                pass
    except (IndexError, KeyError):
        html += f" <span style='color: #00FF00;'>{corrected_segment_text}</span>"
    
    html += "</p></div>"
    return html

# =========================
# HÀM XỬ LÝ CHUNG (ĐÃ SỬA LỖI BẰNG FFMPEG)
# =========================
def process_audio(audio_source_name, audio_bytes, use_gemini_flag, api_key, dia_pipeline, suffix=".wav"):
    
    if not whisper_model:
        st.error("Model Whisper chưa sẵn sàng.")
        return
    if not dia_pipeline:
        st.error("Model Phân biệt người nói (pyannote) chưa sẵn sàng.")
        return

    tmp_path_in = None
    tmp_path_wav = None
    
    try:
        with st.spinner(f"Đang xử lý {audio_source_name}..."):
            
            # --- BƯỚC 1: TẠO FILE TẠM GỐC ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_path_in = tmp_in.name

            # --- BƯỚC 2: CHUYỂN ĐỔI SANG WAV BẰNG FFMPEG (Giải pháp dứt điểm) ---
            st.info("⏳ Bước 1/5: Chuẩn hóa âm thanh (FFmpeg)...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
                tmp_path_wav = tmp_out.name
            
            try:
                cmd = [
                    "ffmpeg",
                    "-i", tmp_path_in,
                    "-ar", "16000",       # Resample to 16kHz
                    "-ac", "1",           # Set to 1 channel (mono)
                    "-map_metadata", "-1",
                    "-fflags", "+genpts",
                    "-y",                 # Overwrite output file
                    tmp_path_wav
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True) # Thêm text=True
            except subprocess.CalledProcessError as e:
                st.error(f"Lỗi khi chạy FFmpeg để chuyển đổi file: {e.stderr}", icon="🔥")
                return # Dừng lại nếu không chuyển đổi được

            # --- BƯỚC 3: TẢI AUDIO (Giờ dùng file wav đã chuẩn hóa) ---
            st.info("⏳ Bước 2/5: Tải file âm thanh (torchaudio)...")
            try:
                waveform, sample_rate = torchaudio.load(tmp_path_wav)
            except Exception as e:
                st.error(f"Lỗi khi đọc file WAV đã chuyển đổi: {e}", icon="🔥")
                return

            # --- BƯỚC 4: CHẠY PHÂN BIỆT NGƯỜI NÓI ---
            st.info("⏳ Bước 3/5: Phân biệt người nói (pyannote)...")
            audio_data = {'waveform': waveform, 'sample_rate': sample_rate}
            diarization = dia_pipeline(audio_data)
            
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append((turn.start, turn.end, speaker))

            # --- BƯỚC 5: CHẠY NHẬN DẠNG (Whisper) ---
            st.info("⏳ Bước 4/5: Nhận dạng giọng nói (Whisper)...")
            segment_list = run_whisper(whisper_model, tmp_path_wav, beam_size)

            if not segment_list:
                st.warning("⚠️ Whisper không phát hiện được giọng nói.")
                return

            # --- BƯỚC 6: MERGE, SỬA LỖI VÀ HIỂN THỊ ---
            st.info("⏳ Bước 5/5: Gán nhãn, sửa lỗi và hiển thị...")
            
            st.markdown("### Kết quả gốc (Whisper) với Timeline")
            original_container = st.container(height=300)

            st.markdown("### Kết quả đã sửa (Gemini) với Timeline")
            corrected_container = st.container(height=300)
            
            original_html_full = ""
            corrected_html_full = ""
            all_raw_text = []
            all_corrected_text = []
            gemini_key_ok = (api_key is not None)

            for segment in segment_list:
                segment_midpoint = (segment.start + segment.end) / 2
                assigned_speaker = "UNKNOWN"
                for start, end, speaker in speaker_turns:
                    if start <= segment_midpoint <= end:
                        assigned_speaker = speaker
                        break
                
                raw_text = segment.text.strip()
                all_raw_text.append(raw_text)
                
                # --- Xây dựng HTML Gốc ---
                seg_start_f = format_timestamp(segment.start)
                seg_end_f = format_timestamp(segment.end)
                original_html = f"<div style='background-color:#222; border-left: 3px solid #FFD700; padding: 10px; border-radius: 5px; font-family: monospace; margin-bottom: 5px;'>"
                original_html += f"<p style='margin-bottom: 5px;'><strong style='color: cyan;'>[{assigned_speaker}]</strong> <strong>[{seg_start_f} -> {seg_end_f}]</strong>"
                
                if segment.words:
                    for word in segment.words:
                        word_start = format_timestamp(word.start)
                        word_end = format_timestamp(word.end)
                        original_html += f" <span title='{word_start} -> {word_end}' style='cursor: help;'>{word.word}</span>"
                else:
                    original_html += f" {segment.text}"
                original_html += "</p></div>"
                original_html_full += original_html
                
                # --- XỬ LÝ GEMINI (NẾU BẬT) ---
                if use_gemini_flag and gemini_key_ok:
                    corrected_text = correct_spelling_with_gemini(raw_text, api_key)
                    all_corrected_text.append(corrected_text)
                    corrected_html = build_corrected_timeline_html(segment, assigned_speaker, corrected_text)
                    corrected_html_full += corrected_html
                else:
                    all_corrected_text.append(raw_text)
                    corrected_html_full += original_html 

            # Hiển thị kết quả (Batch)
            original_container.markdown(original_html_full, unsafe_allow_html=True)
            corrected_container.markdown(corrected_html_full, unsafe_allow_html=True)

            st.success("🎉 Hoàn thành xử lý toàn bộ file!")

            # Ghi log (sau khi đã xử lý hết)
            final_raw = " ".join(all_raw_text)
            final_corrected = " ".join(all_corrected_text)
            
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(f"--- [Nguồn: {audio_source_name} | {time.ctime()}] ---\n")
                log_file.write(f"[Gốc] {final_raw}\n")
                if use_gemini_flag and final_raw != final_corrected:
                    log_file.write(f"[Sửa] {final_corrected}\n")
                log_file.write("\n")

    # DÒNG 348 CỦA BẠN LÀ DÒNG NÀY
    except Exception as e: 
        st.error(f"❌ Lỗi khi xử lý âm thanh: {e}")
    
    finally:
        # Xóa CẢ HAI file tạm
        if tmp_path_in and os.path.exists(tmp_path_in):
            os.remove(tmp_path_in)
        if tmp_path_wav and os.path.exists(tmp_path_wav):
            os.remove(tmp_path_wav)

# =========================
# GIAO DIỆN CHIA THEO TAB (ĐÃ SỬA: truyền suffix)
# =========================
tab1, tab2 = st.tabs(["📁 Tải file lên", "🔴 Ghi âm trực tiếp"])

with tab1:
    st.header("Tải file âm thanh")
    uploaded_file = st.file_uploader("Chọn tệp âm thanh", type=["wav", "mp3", "m4a"], label_visibility="collapsed")
    
    if uploaded_file is not None and whisper_model:
        audio_bytes = uploaded_file.read()
        
        _ , file_suffix = os.path.splitext(uploaded_file.name)
        
        process_audio(f"File: {uploaded_file.name}", audio_bytes, use_gemini, gemini_api_key, diarization_pipeline, suffix=file_suffix)

with tab2:
    st.header("Ghi âm từ Micro")
    st.markdown("Nhấn nút bên dưới để bắt đầu ghi âm. Nhấn lần nữa để dừng.")
    
    audio_bytes = audio_recorder(
        text="Nhấn để ghi âm",
        recording_color="#e84040",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="3x",
    )
    
    if audio_bytes and whisper_model:
        st.audio(audio_bytes, format="audio/wav")
        process_audio("Ghi âm trực tiếp", audio_bytes, use_gemini, gemini_api_key, diarization_pipeline, suffix=".wav")

# --- NÚT TẢI LOG (LUÔN HIỂN THỊ) ---
if os.path.exists(log_filename):
    with open(log_filename, "r", encoding="utf-8") as f:
        log_data = f.read()
    st.sidebar.download_button("📥 Tải toàn bộ log.txt", 
                               data=log_data, 
                               file_name=log_filename)