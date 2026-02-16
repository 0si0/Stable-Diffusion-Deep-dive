# Stable-Diffusion-Deep-dive

원래 Stable Diffusion은 간단하게 Pipeline 을 활용하여

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision = "fp16", torch_dtype = torch.float16, use_auth_token = True).to("cuda")

image = pipe("An astronaut scuba Diving").images[0]

이렇게 간단히 할 수 있지만, 그 속의 세부적인 작동을 직접 코드로 구현해보았다.



Stable Diffusion은 일반적인 DIffusion 모델과 다른 특징들이 여럿 있다.

Stable Diffusion은 크게 2가지의 핵심 component로 구성되어 있다. 

우선 Text Understander로 사용자가 입력한 prompt를 이해해야 하며,
그를 바탕으로 Image Generator가 작동하여 그에 맞는 이미지를 출력한다.


<img width="1862" height="746" alt="image" src="https://github.com/user-attachments/assets/109d30c8-8567-4a97-9c97-e0a04643c412" />


Image Generator는 또 두 부분으로 나뉜다.

<img width="1788" height="668" alt="image" src="https://github.com/user-attachments/assets/f8b419a3-b459-472e-90be-50da43c1bf39" />


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

### Classifier Free Guidance

prompt를 아무리 입력해도, 모델이 여전히 prompt보다 default noisy input image에 의존하여 이미지를 생성할 수도 있다. 따라서 이럴 때 더 prompt에 가깝게 하기 위해서 CFG를 활용한다.

CFG는 prompt를 사용하여 만든 이미지인 conditional sampling과,
prompt 없이 만든 이미지인 unconditional sampling을 각각 만들어서,
두개의 오차만큼 더 학습시켜서 conditional sampling에 더 가깝도록 이미지를 생성할 수 있게
조정하는 방법이다.


### How diffusion works

Diffusion은 항상 랜덤한 noise를 더해준다.

<img width="1182" height="636" alt="image" src="https://github.com/user-attachments/assets/43e78bd0-0df6-499f-bf45-015c40977143" />


Image 하나를 고른 뒤, random noise를 생성하고, 이만큼 image에 추가하여 dataset을 구축한다.

<img width="1182" height="656" alt="image" src="https://github.com/user-attachments/assets/7a70d4d6-6509-49a3-a049-742c03b57202" />

노이즈가 섞인 이미지들이 들어있는 training dataset을 Unet에 넣고,
Unet이 어떤 노이즈가 포함되어있는지를 예측해서 Unet Prediction을 내놓으면,
Actual noise와 얼마나 차이가 나는지 Loss Fuction으로 계산하여 그만큼 차이를 보정한다.

<img width="1182" height="656" alt="image" src="https://github.com/user-attachments/assets/c2d83962-cea2-4385-b615-5a95f7569d98" />

이제 step을 계속 거치며, 예측된 noise만큼을 이미지에서 빼면서 원본 이미지를 차근차근히
예측해 나간다. 
이것이 Diffusion이다.

### **Diffusion on Compressed (Latent) Data Instead of the Pixel Image**

지금까지는 전체 픽셀을 기준으로 노이즈를 삽입했기때문에 시간이 좀 걸렸다.

지금부터는 Compressed Data, 즉 압축된 이미지를 바탕으로 Diffusion을 진행할 것이다.

<img width="1182" height="662" alt="image" src="https://github.com/user-attachments/assets/12a5148b-3c2b-415b-ae48-180dbb1fed2b" />




