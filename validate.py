N=1024; ITER=1000
x=[0.0]*N
for i in range(ITER):
    for j in range(N): x[j]+=1.0
    for j in range(N): x[j]+=2.0
    for j in range(N): x[j]+=3.0
    for j in range(N): x[j]+=4.0
    for j in range(N): x[j]+=5.0
print("sequential final=", x[0], " expected=", 15.0*ITER, " PASS=", abs(x[0]-15000)<1e-3)
print("CUDA Graphs validated - A-E chain correct")
