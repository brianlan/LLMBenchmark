"""evalscope 跑在子进程里，没法从 run.py 打补丁，只能靠 sitecustomize（PYTHONPATH 指到这里）。

nltk 从 CVE-2026-12926 起给 edit_distance 加了 2000 字符的 DoS 上限，而 OCRBench-v2 的
「整页 OCR」答案动辄几千字符，直接 ValueError 打挂整轮评测。被比较的是我们自己模型吐的文本，
可信，放开上限。
"""

try:
    from nltk.metrics import distance as _distance  # 'import nltk.metrics.distance as x' 这种写法会报错

    _distance.MAX_DISTANCE_INPUT_LEN = 20000
except Exception:  # nltk 没装就算了，不是所有 benchmark 都用得上
    pass
