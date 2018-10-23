g++ init.cpp -o init.so -fPIC -shared -pthread -O3 -march=native
g++ test.cpp -o test.so -fPIC -shared -pthread -O3 -march=native
g++ valid.cpp -o valid.so -fPIC -shared -pthread -O3 -march=native