import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from diffusers import StableDiffusion3Pipeline
from modelscope import snapshot_download

class Audio2Image:
    def __init__(self,):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")   # 用于ASR等，32维
        model_id = snapshot_download('AI-ModelScope/stable-diffusion-3.5-large')
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        self.pipe = pipe
    def audio2text(self,audio_path):
        audio_input, sample_rate = sf.read(audio_path)  # (31129,)
        input_values = self.processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values  # torch.Size([1, 31129])

        logits = self.model(input_values).logits     # torch.Size([1, 97, 32])
        predicted_ids = torch.argmax(logits, dim=-1)    # torch.Size([1, 97])

        transcription = self.processor.decode(predicted_ids[0])
        return transcription
    
    def text2image(self,transcription,savepath):
        image = self.pipe(transcription,num_inference_steps=28,guidance_scale=3.5).images[0]
        image.save(savepath)

    def main(self,audio_path,savepath):
        transcription = self.audio2text(audio_path)
        self.text2image(transcription)

if __name__ == "__main__":
    audio_path = "sample.wav"
    savepath = "./res.jpg"
    api = Audio2Image()
    api.main(audio_path,savepath)
