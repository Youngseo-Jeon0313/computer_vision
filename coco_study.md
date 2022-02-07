coco - Microsoft Common Objects in Context


Attention is all you need 논문 리뷰  

Why?  

Long-term dependency problem과 parallelization 문제 때문에 Transformer가 대두되었다. 이전 모델은 많은 개선점이 있음에도 long-term dependency problem이 있다는 단점이 있었다. 만약 “저는 대학원에 가는 것이 행복하고, 햇살은 따뜻하고, 가서 딥러닝을 공부할 생각을 하니 기분이 좋다” 라는 문장에서 대학원과 가서 딥러닝은 밀접한 관계를 가져, NLP 처리에 중요한 단서이나, 이전의 recurrent model을 사용한다면, 이 ltdp문제로 정보가 소실되어 거리가 멀 때 해당정보를 이용하지 못할 수 있다. 그래서 attention을 사용한다.  

Parallelization 문제는 n번째 hidden state 학습 시 전 단계의 hidden start가 필요하기 때문에 순차적으로 계산이 되어야만 한다. 즉 병렬적인 계산을 할 수 없어, 컴퓨터 입장에서 굉장히 비효율 적이며 계산속도가 느렸다. 하지만 transfomer에서는 encoder에서 각각 position에 대해 attetion만 해주면 되고, decoder에선 masking으로 병렬 처리가 가능해졌다.  

Transformer-Attention Architecture  

일단 먼저 간단하게만 설명하자면, 왼쪽에 보이는 연결선처럼 문장 속 다양한 관계들을 나타내기 위해 이 architecture가 반영되었다고 생각하면 됩니다. 크게 encoder와 decoder로 나뉘는데, 자세한 프레임워크는 뒤에서 설명하도록 하겠습니다. 크게 보았을 때 이 회색 부분이 N번 반복되는 회귀형을 따르며, 이렇게 반복되는 동안의 output들이 다시 추가적으로 input으로 들어간다고 보면 됩니다.  

Input은 query / key / value 3개이고, output은 이 세 개의 가중합으로 계산됩니다. 그렇게 나온 output을 그림으로 표현했을 때 왼쪽과 같다고 보면 되는데, 쉽게 말해 우리가. ‘making ~ more difficult’라는 구가 있다고 했을 때 making이라는 단어가 이 세 단어로 이루어진 문장 속에서 중요한 역할을 한다. 라는 관계적인 것의 확률값이 encoding과 decoding을 거쳐 나오게 됩니다. 이렇게 학습이 되면 이후 문장을 넣었을 때 Architecture에 따라 문장을 번역하게 됩니다.  

그럼 이제 더 자세히 들여다보도록 하겠습니다.  

Positional encoding  

우선 맨 오른쪽에 빨간 원으로 표시한 것처럼positional encoding 단계를 수행하는데, 그 이유는 RNN에서는 연속된 데이터를 순서대로 집어넣어줘서 순서 정보를 따로 안 넣어줘도 됐습니다. 근데 병렬화를 위해 어텐션 모델에서는 데이터를 한번에 집어넣어 문장의 순서 정보를 표시해주는 것이 필요해졌어요. 그래서 이제 sin, cos 조합으로 단어 간의 상대적인 위치 데이터를 더해주게 됩니다.   

Encoder  

다음은 인코더 입니다. 트랜스포머는 하이퍼파라미터인 num_layers 개수 만큼 인코더 층을 쌓습니다. 논문에서는 6개의 층을 사용합니다. 이 인코더에선 셀프어텐션과 피드포워드 신경망으로 크게 두개의 서브레이어로 나누어 집니다. 

Attention 

논문에선 제목처럼 self-attention을 핵심인데, 이 self-attention이라는 것은 어텐션을 자기 자신에게 수행한다는 의미입니다. 어텐션함수는 쿼리에 대해 모든 키와 유사도를 각각 구하고, 이 유사도를 가중치로 하여 키와 맵핑된 각각의 값에 반영해줍니다. 반영된 값을 모두 가중합하여 리턴합니다. 

다음처럼 디코더 셀의 은닉상태가 쿼리고 인코더 셀의 은닉 상태가 키라는 점에서 서로 다른 값을 가졌으나, 셀프 어텐션에서는 쿼리,키,밸류가 모두 동일합니다. 

Encoder  

연관성을 파악하기 전 Q,K,V 벡터를 얻어야 하는데, 초기입력의 차원을 가지는 단어로 셀프어텐션을 수행하지 않고, 각 단어 벡터로부터 Q,K,V벡터를 얻는다. 이 벡터는 초기 입력 차원보다 더 작은 차원을 받는다. 사진을 보아도 더 작은 차원으로 벡터가 변환이 되었음을 알 수 있다. 논문에서 가중치 행렬의 크기는 다음과 같은 크기를 가집니다. 이 가중치 행렬들은 훈련하며 학습이 됩니다. 

Encoder-Scaled dot product Attention 

이 스케일드 닷 프로덕트 어텐션을 간단히 설명하자면, 단어간의 관계를 파악하기 위한 함수입니다. 우리는 전단계로 각각의 Q,K,V를 얻었는데, 각 Q벡터는 모든 K에 대해 어텐션 스코어(즉 관계도)를 구하고 어텐션 분포를 구한 후 모든 V벡터를 가중합하여 어텐션값을 구하게 됩니다. 예시를 들어 한번 설명을 해보겠습니다. 

I를 기준으로 설명을 드릴텐데, 역시 I뿐만 아니라 모든 단어에 대해 다음과 같이 계산을 해주어야 한다. 단어 I에 대한 Q 벡터가 모든 K벡터에 대한 어텐션 스코어를 구하는 것을 보이는데, 위의 어텐션 스코어에서 K벡터의 차원을 나타내는 디멘션 k에 루트를 씌워서 스케일링해주었습니다. 다음의 값에 어텐션 분포를 구하고, 각 V벡터와 가중합하여 어텐션값을 도출해냅니다. 

물론 이렇게 각 단어마다 따로 할 필요는 없습니다. 한 번에 병렬 연산으로 수행해 빠르게 계산을 끝낼 수 있습니다. 다음식을 보면 논문의 수식과 정확히 일치함을 확인할 수 있습니다. 

Multi-head Attention 

방금 저희는 굳이 행렬 곱을 하여서 Q,K,V의 벡터로 바꾸어서 어텐션을 수행하였는데, 왜 입력된 차원을 사용하지 않고 굳이 차원을 축소시킨 벡터로 어텐션을 수행했는지에 대해 알아보겠습니다. 이 Multi-head Attention은 Encoder의 첫번째 sub-layer입니다. 논문에서는 어텐션을 병렬로 사용하는 것에 대해 계산에 효과적이라고 생각하였고, 한 문장을 여러 시각에서 볼 수 있어, 문장의 내용에서 놓치는 점이 없게 하겠다. 즉 여러 시각으로 정보를 수집하겠다는 생각으로 head를 여럿으로 나누었다. 여기서 중요한 점은 인코더의 입력으로 들어온 행렬의 크기가 아직 유지됨입니다. 여러 개의 인코더를 거치기 때문에 계속 동일 크기로 유지되어야 다시 입력이 될 수 있습니다. 멀티 어텐션 헤드에서 각각 병렬을 수행후 다른 어텐션 헤드들을 연결하여 이처럼 유지될 수 있습니다. 

Add & Norm 

Residual connection은 서브층의 입력과 출력을 더하는 것을 말한다. 앞서 언급했듯이 서브층의 입력과 출력이 동일한 차원을 가지므로 덧셈 연산이 가능하다. 

Layer Noimalization은 덧셈 연산을 거친 후 정규화 과정을 거치게 되는데, 텐서의 마지막 차원에서 평균과 분산을 구하고, 어떤 수식을 통해 값을 정규화 하여 학습을 돕게 됩니다. 여기서 텐서의 마지막 차원은 트랜스포머에서 모델의 디멘션을 의미하겠죠. 

keras에서 층 저 

Decoder  

Encoder와 마찬가지로 N개의 layer로 구성되어 있다.  

@@셀프 어텐션에서 Q,K,V가 전부 동일하다.  

SL을 이용해 학습되므로, 학습과정에서 번역할 문장에 해당되는 문장 전체를 한번에 입력받는다. 역시 positional encoding을 적용한 후에 문장 matrix가 적용이 된다.  

Decoder의 첫번째 sub layer는 encoder와 동일한 self-attention이지만 추가적으로 encoder와 달리 masking을 해주는데, 입력 vector는 전체 문장으로 input으로 되기 때문에 모든 값을 참조 가능하지만, 이 masking을 통해 position이후에 position에 attetion을 주지 못하게 해서 예측을 하지 못하게 한다. 만약 “I am a student”라는 문장을 번역 시 “I am”뒤 올 단어를 예측하려면 “a”, “student”에 대한 정보를 얻을 수 없게 해야 하는 것이다.   

이 masked decoder를 사용하기 위해 뒤의 query와 key값을 -INF로 만들어 주어, softmax 계산시 0이 나오게 한다.  

이후 masked self attention layer의 output vector는 encoder block과 동일학 residual block과 layer normalization을 거친 후 encoder-decoder attetion과정을 거친다.  

이 encoder-decoder layer에서는 맨 마지막 encoder block에서 출력된 key, value 행렬로 self attention 매커니즘을 한번더 수행한다. Decoder의 과정은 self attention이 아니다. Self attention의 경우 query,key,value가 같은 input으로 생성되지만 이 layer는 query는 decoder 행렬이지만 key와 value는 encoder 행렬로 사용한다.  

모든 encoder와 이전 decoder block을 거친 vector는 최상단 layer인 linear와 softmax를 차례로 거친다. 각각 Linear Layer는 단순한 FC로 마지막 decoder block의 vector를 logit vecotr로 변환하며, softmax layer는 이 logit vector를 각 token이 위치할 확률로 바꾸어 준다.  

  

Self attention 계산단계  

Self attention 계산단계를 구체적으로 설명해보자면, 이렇게 단계적으로 말할 수 있다. 일단 각 단어에 해당하는 Query, Key, Value 값을 각각 벡터로 생성한다. 이후 단어별 score을 계산하는데, 특정 위치에서 단어, 즉 query가 다른 단어들을 얼마나 attention할거니?를 계산한다고 이해하면 된다. 이후 밑에 있는 이 함수를 이용해서 Q,K,V를 계산하여 output으로 ratio를 내보낸다. 오른쪽 그림은 벡터를 그림으로 나타낸 것이다.  

  

코드 엿보기  

마지막으로 관련 코드 결과값을 살짝 엿보면, 왼쪽은 우리가 training시킨 후 각각의 독일어 단어와 영어 단어의 상관관계성을 색으로 나타낸 그래프입니다. 밝은 색일수록 상관관계가 높음을 뜻합니다. 또한 이 단어에서의 순서를 보면, 독일어는 이렇게 쭉 나열되어있고, 이것에 따라서 영어는 각각의 단어의 연관성 그리고 단어의 의미에 따라 이렇게 나열이 되었다라는 것을 표현함을 알 수 있습니다.   

오른쪽은 이제 test set에 있는 독일어 단어를 번역한 결과 예측값과 정답값을 비교하여 블루 스코어값을 낸 것인데, 보면 그래도 잘 번역했음을 알 수 있습니다.  

  

마지막으로 후속 공부 설명,, 
