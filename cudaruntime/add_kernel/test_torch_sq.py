# test_square_add.py
import torch
import square_add  # ← 这就是你编译出来的模块！

# 创建测试数据
a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')

# 调用自定义算子
c = square_add.square_add(a, b)

# 强制同步，看到 printf 输出！
torch.cuda.synchronize()

print("First 5 elements:")
print("a[:5] =", a[:5])
print("b[:5] =", b[:5])
print("c[:5] =", c[:5])
print("Check: a² + b² ≈ c?")
print((a[:5]**2 + b[:5]**2).cpu().numpy())