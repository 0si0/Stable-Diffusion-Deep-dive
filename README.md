# Stable-Diffusion-Deep-dive

원래 Stable Diffusion은 간단하게 Pipeline 을 활용하여

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision = "fp16", torch_dtype = torch.float16, use_auth_token = True).to("cuda")

image = pipe("An astronaut scuba Diving").images[0]

이렇게 간단히 할 수 있지만, 그 속의 세부적인 작동을 직접 코드로 구현해보았다.


