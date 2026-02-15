# Stable-Diffusion-Deep-dive

원래 Stable Diffusion은 간단하게 Pipeline 을 활용하여

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision = "fp16", torch_dtype = torch.float16, use_auth_token = True).to("cuda")

image = pipe("An astronaut scuba Diving").images[0]

이렇게 간단히 할 수 있지만, 그 속의 세부적인 작동을 직접 코드로 구현해보았다.



Stable Diffusion은 하나의 독립적인 실체가 아니라, 크게는 두가지 component로 이루어졌다고 볼 수 있다.

우선 Text Understander로 사용자가 입력한 prompt를 이해해야 하며,
그를 바탕으로 Image Generator가 작동하여 그에 맞는 이미지를 출력한다.

![스크린샷 2026-02-01 오후 9.33.20.png](attachment:23b4ba12-c6cd-4b22-b76c-3e9b0cc1cd65:스크린샷_2026-02-01_오후_9.33.20.png)

Image Generator는 또 두 부분으로 나뉜다.

![스크린샷 2026-02-01 오후 9.36.30.png](attachment:c14caf94-ff0b-4154-b6e6-ec68d034023f:스크린샷_2026-02-01_오후_9.36.30.png)

따라서 총 3가지 component 각각을 살펴보겠다.

- **ClipText** for text encoding.
    
    Input: text.
    
    Output: 77 token embeddings vectors, each in 768 dimensions.
    
- **UNet + Scheduler** to gradually process/diffuse information in the information (latent) space.
    
    Input: text embeddings and a starting multi-dimensional array (structured lists of numbers, also called a *tensor*) made up of noise.
    
    Output: A processed information array
    
- **Autoencoder Decoder** that paints the final image using the processed information array.
    
    Input: The processed information array (dimensions: (4,64,64))
    
    Output: The resulting image (dimensions: (3, 512, 512) which are (red/green/blue, width, height))
    

## Stable Diffusion core component

### VAE (Variational AutoEncoder)

인코더 = 원본 이미지를 압축해서 latent image로 만듦

디코더 = latent → 픽셀 이미지로 복원

### Unet

Diffusion model의 핵심 네트워크.

- 입력: 현재 latent + 현재 timestep + 텍스트 임베딩(조건)
- 출력: “노이즈를 얼마나 제거해야 하는지(노이즈 예측)” 같은 값을 예측

### CLIP

- Clip Tokenizer = 프롬프트 문자열(예: `"a cat"`)을 **토큰(token)** 으로 쪼개고, 각 토큰을 **정수 ID**로 바꿈.
- Clip TextModel = 위 토큰 ID를 입력으로 받아서, 모델이 이해할 수 있는 **텍스트 임베딩(벡터)** 로 바꿔줌.

### Scheduler

“샘플링 루프에서 latent를 어떻게 업데이트할지” 정해주는 **스케줄러**.

![스크린샷 2026-02-04 오후 8.54.29.png](attachment:c733d07c-8270-4ca2-b21d-65b3ef265984:스크린샷_2026-02-04_오후_8.54.29.png)

Diffusion은 Image Informaton Creator 안에서 일어나는 과정이다.

Text Encoder가 출력해낸 Token embeddings를 input으로 받고, 
Random image information tensor에서 시작해서, 여러 스텝을 거쳐가며 
사용자가 원하는 image information tensor를 제작해 나간다.

### How diffusion works

Diffusion은 항상 랜덤한 noise를 더해준다.

![스크린샷 2026-02-04 오후 9.37.18.png](attachment:7050046f-fbe4-417f-809e-9308f9db1cbe:스크린샷_2026-02-04_오후_9.37.18.png)

Image 하나를 고른 뒤, random noise를 생성하고, 이만큼 image에 추가하여 dataset을 구축한다.

![스크린샷 2026-02-04 오후 9.38.08.png](attachment:446153c2-ada2-4828-9bb1-f00f3520c46e:스크린샷_2026-02-04_오후_9.38.08.png)

노이즈가 섞인 이미지들이 들어있는 training dataset을 Unet에 넣고,
Unet이 어떤 노이즈가 포함되어있는지를 예측해서 Unet Prediction을 내놓으면,
Actual noise와 얼마나 차이가 나는지 Loss Fuction으로 계산하여 그만큼 차이를 보정한다.

![스크린샷 2026-02-04 오후 9.46.47.png](attachment:ac1cdb96-d597-47f8-880c-78da2c1ff805:스크린샷_2026-02-04_오후_9.46.47.png)

이제 step을 계속 거치며, 예측된 noise만큼을 이미지에서 빼면서 원본 이미지를 차근차근히
예측해 나간다. 
이것이 Diffusion이다.

### **Diffusion on Compressed (Latent) Data Instead of the Pixel Image**

지금까지는 전체 픽셀을 기준으로 노이즈를 삽입했기때문에 시간이 좀 걸렸다.

지금부터는 Compressed Data, 즉 압축된 이미지를 바탕으로 Diffusion을 진행할 것이다.

![스크린샷 2026-02-04 오후 9.52.59.png](attachment:34e83283-0dac-4e0c-812d-c9247f1b11c4:스크린샷_2026-02-04_오후_9.52.59.png)
