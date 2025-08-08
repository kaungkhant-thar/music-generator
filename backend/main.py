import modal
import os

app = modal.App("music-generator")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        [
            "git clone git@github.com:ace-step/ACE-Step.git /tmp/ACE-Step",
            "cd /tmp/ACE-Step && pip install .",
        ]
    )
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music_gen_secrets")


@app.function(secrets=[modal.Secret.from_name("music-gen-secrets")])
def f():
    print("KKT")
    print(os.environ["test"])


@app.local_entrypoint()
def main():
    f.remote()


# @app.cls(
#     image=image,
#     gpu="L40S",
#     volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
#     secrets=[music_gen_secrets],
#     scaledown_window=15,
# )
# class MusicGenServer:
#     tokenizer: PreTrainedTokenizerBase
#     llm_model: AutoModelForCausalLM
#     music_model: ACEStepPipeline
#     image_pipe: AutoPipelineForText2Image

#     @modal.enter()
#     def load_model(self):
# from acestep.pipeline_ace_step import ACEStepPipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
# from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
# import torch

#         self.music_model = ACEStepPipeline(
#             checkpoint_dir="/models",
#             dtype="bfloat16",
#             torch_compile=False,
#             cpu_offload=False,
#             overlapped_decode=False,
#         )

#         model_id = "Qwen/Qwen2-7B-Instruct"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)

#         self.llm_model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype="auto",
#             device_map="auto",
#             cache_dir="/.cache/huggingface",
#         )

#         self.image_pipe = AutoPipelineForText2Image.from_pretrained(
#             pretrained_model_or_path="stabilityai/sdxl-turbo",
#             torch_dtype=torch.float16,
#             variant="fp16",
#             cache_dir="/.cache/huggingface",
#         )
#         self.image_pipe.to("cuda")
