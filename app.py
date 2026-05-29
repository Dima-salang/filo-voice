
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model, KokoroTTSOptions
import ollama

# Initialize the speech-to-text (Moonshine) and text-to-speech (Kokoro) engines
# FastRTC handles downloading these tiny, highly optimized variants automatically on first boot
stt_model = get_stt_model()
tts_model = get_tts_model()

# Keep track of the chat context so the AI remembers what you just said
chat_history = [
    {"role": "system", "content": "You are a helpful, brief, and friendly voice assistant. Keep answers concise so they sound natural when spoken aloud."}
]

def voice_chat_loop(audio):
    global chat_history
    
    # 1. Convert your spoken audio into text using Moonshine
    user_prompt = stt_model.stt(audio)
    if not user_prompt.strip():
        return
        
    print(f"🗣️ You: {user_prompt}")
    chat_history.append({"role": "user", "content": user_prompt})
    
    # 2. Feed the text to your local Ollama server running Llama 3.2
    response = ollama.chat(
        model="llama3.2:3b",
        messages=chat_history
    )
    
    ai_response_text = response['message']['content']
    print(f"🤖 AI: {ai_response_text}")
    chat_history.append({"role": "assistant", "content": ai_response_text})

    options = KokoroTTSOptions(
        voice="af_heart",  # Swap this to af_sarah, bf_emma, etc.
        speed=1.0,
        lang="en-us"
    )
    
    # 3. Stream the AI's response text back out loud using Kokoro
    for audio_chunk in tts_model.stream_tts_sync(ai_response_text, options):
        yield audio_chunk

# Set up the real-time audio communication stream
stream = Stream(
    handler=ReplyOnPause(voice_chat_loop), 
    modality="audio", 
    mode="send-receive"
)

if __name__ == "__main__":
    print("🚀 Launching local voice chat panel...")
    # This fires up a local web UI so you can test it directly in your browser
    stream.ui.launch()
