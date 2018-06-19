import numpy as np

np.random.seed(30) #랜덤 초기값 생성, 없으면 매번 데이터가 달라진다.

x0 = np.random.rand(100); #0과 1사이의 100개의 수를 생성해준다 , 이때 넘파이 1차원 배열 형태로 주어진다.
x1 = np.random.rand(100);
y = 3*x0 + 4*x1 + np.random.randn(100)         #학습데이터, 노이즈 추가, randn()은 평균이 0고 분산이 1이게 만들어 줌
                                                #노이즈는 실제로 수식에 의해 계산된 결과를 학습하는것이기 때문에 학습의 효과를 보기위해 일부로 노이즈 추가

f = open("formula1.csv","w") #write 모드로 furmala1 파일을 연다
f.write("x0,x1,y\n")       #속성 이름

for v1,v2,v3 in zip(x0,x1,y):   #zip함수로 반복문을 돌기 가능 x0,x1,y를 하나의 튜플로 만들어서 한다
    f.write(str(v1)+","+str(v2)+","+str(v3)+"\n")
f.close()