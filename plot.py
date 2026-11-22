import sys, matplotlib.pyplot as plt
seq=float(sys.argv[1]) if len(sys.argv)>1 else 12.5
gr=float(sys.argv[2]) if len(sys.argv)>2 else 1.8
plt.bar(["Sequential 5000 launches","Graph 1000 launches"],[seq,gr])
plt.ylabel("Time (ms)"); plt.title("Sequential vs CUDA Graph - 1000x A-E")
plt.savefig("graph_comparison.png",dpi=150); print(f"saved graph_comparison.png seq={seq} graph={gr} speedup={seq/gr:.2f}x")
