# CUDA Parallel Implementation of Floyd Warshall
This is a Master's degree project related to the Advanced Computer Achitecture course at UNIVR (University of Verona). 
The project is based on Venkataraman's et al. to solve the All Pair Shortest Path problem: https://www.researchgate.net/publication/2566528_A_Blocked_All-Pairs_Shortest-Paths_Algorithm

## How to Build and Execute
You need CUDA 10.0 installed on your machine otherwise you have to modify the CMakeLists setting up the approrpiate CUDA version.

> Note: Make sure to set the right gpu architecture for your device in CMakeLists.txt

```cpp
mkdir build
cd build
cmake ..
make
./floydwarshall <file.mtx> <kernel number>
```

Possible kernel numbers:
1) Naive
2) Coalesced
1) Shared
2) Blocked

## Datasets

- https://sparse.tamu.edu/HB/nos4
- https://sparse.tamu.edu/HB/bcsstk08
- https://sparse.tamu.edu/VDOL/spaceStation_10
- https://sparse.tamu.edu/Schenk_IBMNA/c-44
- https://sparse.tamu.edu/Andrianov/net50

⚠️ Dataset valid format: mtx

## Documentation
In docs folder you can find a report (in Italian) of the results achived using different configurations.
In the same directory there are some web resources used as support material for this project.
