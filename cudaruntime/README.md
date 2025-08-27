# 目录树

## getdeviceproperties.cpp
编译方法：
```
方法1：
g++ getdcuinfo.cpp -o getdeviceinfo -I /opt/dtk/cuda/cuda-11/include -L /opt/dtk/cuda/cuda-11/lib64 -l cudart
方法2：
修改该文件的后缀为cu，直接使用nvcc编译
source /opt/dtk/env.sh
source /opt/dtk/cuda/env.sh
nvcc getdcuinfo.cpp -o getdeviceinfo
```
<img src=./image_res/deviceinfo.png style="zoom:100%;" align=middle>