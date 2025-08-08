import modal
import os
from pydantic import BaseModel
import uuid
import base64
import requests
from typing import List
import boto3
from prompts import PROMPT_GENERATOR_PROMPT, LYRICS_GENERATOR_PROMPT


app = modal.App("music-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        [
            "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
            "cd /tmp/ACE-Step && pip install .",
        ]
    )
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music_gen_secrets")


class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False


class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str


class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]


@app.cls(
    image=image,
    gpu="L40S",
    volumes={
        "/models": model_volume,
        "/.cache/huggingface": hf_volume,
    },
    secrets=[music_gen_secrets],
    scaledown_window=15,
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from diffusers import AutoPipelineForText2Image
        import torch

        # Music generation model
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False,
        )

        # LLM Model
        model_id = "Qwen/Qwen2-7B-Instruct"

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # stable diffusion model (for thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        self.image_pipe.to("cuda")

    def prompt_qwen(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.llm_model.device
        )

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids, max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response

    def generate_prompt(self, description: str):
        # insert into template
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        # run llm inference and return that
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        # insert into template
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
        # run llm inference and return that
        return self.prompt_qwen(full_prompt)

    def generate_categories(self, description: str) -> List[str]:
        prompt = f"Based on the following music description, list 3-5 relevant genres or categories as a comma-separated list. For example: Pop, Electronic, Sad, 80s. Description: '{description}'"

        response_text = self.prompt_qwen(prompt)
        categories = [
            category.strip()
            for category in response_text.split(",")
            if category.strip()
        ]
        return categories

    def generate_and_upload_to_s3(
        self,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        seed: int,
        description_for_categorization: str,
    ) -> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics

        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"Prompt: \n{prompt}")

        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.music_model(
            prompt=prompt,
            lyrics=final_lyrics,
            audio_duration=audio_duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            manual_seeds=str(seed),
            save_path=output_path,
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)

        thumbnail_prompt = f"{prompt}, album cover art"
        image = self.image_pipe(
            prompt=thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0
        ).images[0]
        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"

        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

        categories = self.generate_categories(description_for_categorization)

        return GenerateMusicResponseS3(
            s3_key=audio_s3_key, cover_image_s3_key=image_s3_key, categories=categories
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_description(
        self, request: GenerateFromDescriptionRequest
    ) -> GenerateMusicResponseS3:
        # generate prompt
        prompt = self.generate_prompt(request.full_described_song)

        # generate lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)

        return self.generate_and_upload_to_s3(
            prompt=prompt,
            lyrics=lyrics,
            description_for_categorization=request.full_described_song,
            **request.model_dump(exclude={"full_described_song"}),
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_lyrics(
        self, request: GenerateWithCustomLyricsRequest
    ) -> GenerateMusicResponseS3:
        return self.generate_and_upload_to_s3(
            description_for_categorization=request.lyrics,
            **request.model_dump(exclude={"full_described_song"}),
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_described_lyrics(
        self, request: GenerateWithDescribedLyricsRequest
    ) -> GenerateMusicResponseS3:

        # generate lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.described_lyrics)

        return self.generate_and_upload_to_s3(
            lyrics=lyrics,
            description_for_categorization=request.described_lyrics,
            **request.model_dump(exclude={"described_lyrics"}),
        )


@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate_with_described_lyrics.get_web_url()

    request_data = GenerateWithDescribedLyricsRequest(
        prompt="rock, drum, bass",
        described_lyrics="a story about success",
    )
    payload = request_data.model_dump()

    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()

    result = GenerateMusicResponseS3(**response.json())
    print(f"Success: {result.s3_key} {result.cover_image_s3_key} {result.categories}")
