# How to Compile and Execute #

###Make sure to set the right gpu architecture for in CMakeLists.txt

```cpp
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

# Dataset #

https://sparse.tamu.edu/HB/nos4
https://sparse.tamu.edu/HB/bcsstk08
https://sparse.tamu.edu/VDOL/spaceStation_10
https://sparse.tamu.edu/Schenk_IBMNA/c-44
https://sparse.tamu.edu/Andrianov/net50

formato: mtx
