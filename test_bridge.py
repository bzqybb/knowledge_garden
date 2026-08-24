"""Small smoke demo. API credentials are read only from environment variables."""

import tempfile
from pathlib import Path

from core.engine import analyze_frontier
from core.storage import GardenStore


frontier_text = """
在深度残差网络（ResNet）中，通过引入恒等快捷连接，使梯度可以直接穿透层间传播，
缓解深层神经网络反向传播中的梯度消失问题。
"""


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        store = GardenStore(Path(directory) / "demo.db")
        result = analyze_frontier(store, "残差网络与梯度传播", frontier_text)
        print("提取概念：", "、".join(result["concepts"]))
        for card in result["cards"]:
            print(f"\n【{card['concept']}】\n{card['explanation']}")
