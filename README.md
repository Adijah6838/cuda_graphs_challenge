# CUDA Graphs - Workflow Optimization

5 kernels A-E: +1, +2, +3, +4, +5 per iteration. 1000 iterations = 15000 expected.

## Sequential vs Graph
Sequential: 1000 x 5 = 5000 kernel launches, CPU launch overhead per launch (~5-10us each).
Graph: capture A-E once (cudaStreamBeginCapture/EndCapture), instantiate, launch graph 1000x. 1 launch overhead per 5 kernels. GPU work identical, CPU overhead cut ~5x.

## Results (fill from Colab)
Sequential CPU/GPU ms vs Graph CPU/GPU ms + graph_comparison.png bar chart.

## Build / Run
nvcc -O3 -arch=sm_80 main.cu -o main
./main
python plot.py <seq_ms> <graph_ms>
