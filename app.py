# import streamlit as st
# import os
# import tempfile
# from faster_whisper import WhisperModel

# NEON_GREEN = "\033[92m"   # Xanh sáng
# RESET_COLOR = "\033[0m"   # Reset về mặc định

# def transcribe_chunk(model, file_path):
#     segments, info = model.transcribe(file_path, language="en", vad_filter=True)
#     text = "".join([segment.text for segment in segments])
#     return text

# # Giao diện Streamlit
# st.title("🎤 Real-time Voice Transcription (Whisper AI)")
# st.markdown("Upload audio file (.wav, .mp3) để chuyển giọng nói thành văn bản")

# # Chọn kích thước model
# model_size = st.selectbox("Chọn model:", ["base", "small", "medium", "large-v3"], index=3)
# device = "cuda" if st.checkbox("Dùng GPU (CUDA)", value=True) else "cpu"

# # Tải model Whisper
# @st.cache_resource
# def load_model(model_size, device):
#     return WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")

# model = load_model(model_size, device)
# st.success(f"✅ Model `{model_size}` đã sẵn sàng trên {device.upper()}")

# # Upload file audio
# uploaded_file = st.file_uploader("Chọn tệp âm thanh", type=["wav", "mp3", "m4a"])

# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name

#     st.info("⏳ Đang xử lý âm thanh...")
#     transcription = transcribe_chunk(model, tmp_path)

#     if transcription.strip():
#         st.markdown(f"<span style='color:limegreen; font-weight:bold;'>{transcription}</span>", unsafe_allow_html=True)

#         # Ghi log vào file
#         with open("log.txt", "a", encoding="utf-8") as log_file:
#             log_file.write(transcription + "\n")

#         st.download_button("📥 Tải log.txt", data=transcription, file_name="log.txt")
#     else:
#         st.warning("⚠️ Không phát hiện được giọng nói trong đoạn audio.")

#     os.remove(tmp_path)

import streamlit as st
import os
import tempfile
from faster_whisper import WhisperModel

# =========================
# GIAO DIỆN STREAMLIT
# =========================
st.title("🎤 Real-time Voice Transcription (Whisper AI)")
st.markdown("Tải file âm thanh (.wav, .mp3, .m4a) để chuyển giọng nói thành văn bản bằng WhisperModel")

# =========================
# CHỌN MODEL
# =========================
model_size = st.selectbox("Chọn model:", ["tiny", "base", "small", "medium", "large-v3"], index=2)

# =========================
# HÀM TẢI MODEL (CHỈ DÙNG CPU)
# =========================
@st.cache_resource
def load_model(model_size):
    # Dùng CPU + int8 cho nhẹ
    return WhisperModel(model_size, device="cpu", compute_type="int8")

model = load_model(model_size)
st.success(f"✅ Model `{model_size}` đã sẵn sàng trên CPU")

# =========================
# HÀM NHẬN DIỆN ÂM THANH
# =========================
def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, language="en", vad_filter=True)
    text = "".join([segment.text for segment in segments])
    return text.strip()

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader("🎧 Chọn tệp âm thanh", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("⏳ Đang xử lý âm thanh, vui lòng chờ...")

    try:
        transcription = transcribe_chunk(model, tmp_path)
        if transcription:
            st.markdown(f"<p style='color:limegreen; font-weight:bold;'>{transcription}</p>", unsafe_allow_html=True)

            # Ghi log vào file
            with open("log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(transcription + "\n")

            st.download_button("📥 Tải log.txt", data=transcription, file_name="log.txt")
        else:
            st.warning("⚠️ Không phát hiện được giọng nói trong đoạn audio.")
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý âm thanh: {e}")

    os.remove(tmp_path)
