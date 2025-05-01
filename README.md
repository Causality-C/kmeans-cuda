# Kmeans

## Data sources
First download data  (ie. the 65536 datapoint ones)
```
wget https://www.cs.utexas.edu/~rossbach/cs378h/lab/kmeans-sample-inputs/random-n65536-d32-c16.txt
```
and the answers

```
wget https://www.cs.utexas.edu/~rossbach/cs378h/lab/kmeanspp-sample-inputs/random-n65536-d32-c16-answer.txt
```


## Instructions to run 
```
cmake --preset release
cmake --build --preset release
./build/release/KmeansCuda -i data/random-n65536-d32-c16.txt -k 16 -d 32 -s 8675309 -t 0.00001 -g -b data/random-n65536-d32-c16-answer.txt
```