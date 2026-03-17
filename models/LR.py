import torch.nn as nn
import torch.optim as optim

class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)   # a, b, c 全学出来

    def forward(self, x):
        return self.linear(x)           # 输出 logit；后面会接 BCEWithLogits

model = LR()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 2×10⁻² 快速收敛

for epoch in range(300):                # 一般不到 200 次就收敛
    for xb, yb in dl:
        pred = model(xb)                # shape [batch,1] (logit)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# optional: 打印 a, b, c
a, b = model.linear.weight.data[0].tolist()
c    = model.linear.bias.data.item()
print(f"a={a:.3f}, b={b:.3f}, c={c:.3f}")
