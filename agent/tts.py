import re
import time
from abc import ABC
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import requests
import io
import wave
# from rag.utils import num_tokens_from_string


class Base(ABC):
    def __init__(self, key, model_name, base_url):
        pass

    def tts(self, audio):
        pass

    def normalize_text(self, text):
        return re.sub(r'(\*\*|##\d+\$\$|#)', '', text)
    

class ParlerTTS(Base):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained('parler-tts/parler-tts-mini-v1').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('parler-tts/parler-tts-mini-v1')
        self.sampling_rate = 44000
        
    def _convert_audio_to_wav_bytes(self, audio_array):
        """Convert numpy array to WAV format bytes"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(self.sampling_rate)
            wav_file.writeframes((audio_array * 32767).astype('int16').tobytes())
        
        return wav_buffer.getvalue()
    
    def _chunk_audio(self, audio_bytes, chunk_size=4096):
        """Split audio bytes into chunks"""
        return (audio_bytes[i:i + chunk_size] for i in range(0, len(audio_bytes), chunk_size))
    
    def tts(self, text, description="A natural voice with clear articulation and professional tone.", chunk_size=4096):
        """
        Generate speech from text and stream it in chunks.
        
        Args:
            text (str): Text to convert to speech
            description (str): Voice description for the model
            chunk_size (int): Size of audio chunks in bytes
        
        Yields:
            bytes: Chunks of WAV format audio data
        """
        try:
            # Tokenize inputs
            des_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            text_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            # Generate audio
            with torch.no_grad():  # Reduce memory usage
                generation = self.model.generate(input_ids=des_ids, prompt_input_ids=text_ids)
            
            # Convert to numpy array
            audio_arr = generation.cpu().numpy().squeeze()
            # save audio
            # sf.write('audio.wav', audio_arr, self.sampling_rate)
            
            # Convert to WAV format
            wav_bytes = self._convert_audio_to_wav_bytes(audio_arr)
            
            # Stream in chunks
            for chunk in self._chunk_audio(wav_bytes, chunk_size):
                yield chunk
                
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {str(e)}")

    def tts_to_file(self, text, output_path, description="A natural voice with clear articulation and professional tone."):
        """
        Generate speech and save directly to a file.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the WAV file
            description (str): Voice description for the model
        """
        with open(output_path, 'wb') as f:
            for chunk in self.tts(text, description):
                f.write(chunk)


class QwenTTS(Base):
    def __init__(self, key, model_name, base_url=""):
        import dashscope

        self.model_name = model_name
        dashscope.api_key = key

    def tts(self, text):
        from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
        from dashscope.audio.tts import ResultCallback, SpeechSynthesizer, SpeechSynthesisResult
        from collections import deque

        class Callback(ResultCallback):
            def __init__(self) -> None:
                self.dque = deque()

            def _run(self):
                while True:
                    if not self.dque:
                        time.sleep(0)
                        continue
                    val = self.dque.popleft()
                    if val:
                        yield val
                    else:
                        break

            def on_open(self):
                pass

            def on_complete(self):
                self.dque.append(None)

            def on_error(self, response: SpeechSynthesisResponse):
                raise RuntimeError(str(response))

            def on_close(self):
                pass

            def on_event(self, result: SpeechSynthesisResult):
                if result.get_audio_frame() is not None:
                    self.dque.append(result.get_audio_frame())

        text = self.normalize_text(text)
        callback = Callback()
        SpeechSynthesizer.call(model=self.model_name,
                               text=text,
                               callback=callback,
                               format="mp3")
        try:
            for data in callback._run():
                yield data
            yield num_tokens_from_string(text)

        except Exception as e:
            raise RuntimeError(f"**ERROR**: {e}")


class OpenAITTS(Base):
    def __init__(self, key, model_name="tts-1", base_url="https://api.openai.com/v1"):
        if not base_url: base_url = "https://api.openai.com/v1"
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def tts(self, text, voice="alloy"):
        text = self.normalize_text(text)
        payload = {
            "model": self.model_name,
            "voice": voice,
            "input": text
        }

        response = requests.post(f"{self.base_url}/audio/speech", headers=self.headers, json=payload, stream=True)

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")
        for chunk in response.iter_content():
            if chunk:
                yield chunk


class XinferenceTTS:
    def __init__(self, key, model_name, **kwargs):
        self.base_url = kwargs.get("base_url", None)
        self.model_name = model_name
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

    def tts(self, text, voice="中文女", stream=True):
        payload = {
            "model": self.model_name,
            "input": text,
            "voice": voice
        }

        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            headers=self.headers,
            json=payload,
            stream=stream
        )

        if response.status_code != 200:
            raise Exception(f"**Error**: {response.status_code}, {response.text}")

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                yield chunk


def play_audio_sounddevice(tts):
    import sounddevice as sd
    import numpy as np
    
    audio_chunks = []
    for chunk in tts.tts("Hello, world!"):
        # Convert bytes to numpy array
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunks.append(audio_chunk)
    
    # Concatenate all chunks
    audio_data = np.concatenate(audio_chunks)
    sd.play(audio_data, samplerate=44000)
    sd.wait()  # Wait until audio is finished playing
# test parlet_tts
if __name__ == "__main__":
    tts = ParlerTTS()
    play_audio_sounddevice(tts)
