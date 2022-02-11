swin transformer 개념 + 모델 구현
///
Vit는 우리가 앞에서 발표했던 attention 방법을 vision에 대입한 것이다.

과정을 살펴보면,

이미지가 여러 patch로 나누어져 들어가게 됨
Encoder에 linear하게 들어간다.
*Backbone이 CNN이 아니라 Transformer Encoder임
////

Swin transformer가 나오게 된 배경은 다음과 같은데, 일단 swin이라는 것은 swifted window로, 이미지를 특정 방법으로 조작해서 transformer에 넣겠다는 것이다.

오른쪽 이미지로 semantic segmentation을 이해해보자면, 의미론적으로 우리는 이미지들을 각각 이렇게 분류할 수 있고, 이미지들간의 관계도 뭐 ‘사람이 길을 걷고 있다’ 이런식으로 해석할 수 있다. 이 때 컴퓨터에게 이 많은 픽셀들을 전달하기 위해서는 한계가 존재하는데 일단 픽셀은 height * width이므로 2차원의 값들, quadratic한 정보들이 들어가 연산량이 너무 많고, 그리고 이 분홍색 길과 저 분홍색 길의 관계를 잘 보면 픽셀간 너무 멀다. 그 NLP속 token과는 차원이 다른 먼 거리..
그런데 이걸 압축시키고도 좀 그런데, 왜냐면 사진이라는 건 해상도가 있기 때문에 그걸 또 깨기가 애매함.

지엽적인 정보를 써서 멀리 있는 픽셀 간 관계는 잘 파악하지 못하는 CNN을 사용한 vit와는 다르게 transformer를 대입했다고 생각하면 된다.

////
그래서 swin transforme가 대입이 되었는데 일단 가장 큰 특징은 hierarchical feature map을 사용한다는 것이다. 그 방법을 그냥 간단하게 설명하자면 이미지들을 small sized patch로 나눈 후에 그걸 서로 이웃시켜 transformer layers에 차곡차곡 넣는다.
이 방법에 대해서는 이후에 더 설명하도록 하고, 이 때 효과적으로 FPN이나 U-Net과 같은 feature pyramid networks효과를 낼 수 있는데 이 효과를 살짝 설명해보자면![image](https://user-
////
FPN은 CNN을 사용하는 object detection 모델로, 피라미드 구조를 이용한다. 화면을 보면 피라미드 구조의 위쪽으로 갈수록 low resolution, low-level feature을 이용(즉, CNN features의 왼쪽에 있는 걸로, 가장자리, 곡선 같은 저수준 특징을 알아냄. 그리고 아래쪽으로 갈수록은 약간 질감, 물체의 작은 부분 등의 class를 추론할 수 있음

////
이 계층적인 것을 이 stage를 이용해서 실현한다.
Stage 하나에는 linear embedding과 transformer block에 들어가 이 오른쪽에 있는 이 함수들을 모두 통과한다.
일단 patch partition을 통해 그림을 나눈다. 그 4로 나눈다고 했으니까 이렇게 이제 들어가게 되고 이후에는 이제 사용자가 정의한 대로 

////

Swin transformer는 저 해당 widnow에 대해서만 transformer가 적용되고, 이후 점점 window가 merging됨 

////

W-MSA에서는 아까처럼 4개를 기준으로 window를 나누고 SW-MSA에서는 윈도우를 2,2로 shift함 이러면 이제 윈도우 사이의 연결성을 나타낼 수가 있음. 그래서 아까 보면 윈도우 내에서만 attention이 적용되었는데, 계속 이 과정 속에서도 윈도우 내에서만 attentio을 시켜도 연결성이 어느정도 해결됨

