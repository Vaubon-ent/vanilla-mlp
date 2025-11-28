13M Params:
    On CPU computing(static LR):
    - 10 epocs -> 5% precision
    - 50 epocs -> 11,35% precision

    on CPU computing (dynamic LR):
    - 10 epocs -> 12,33% (exec time: 525s)
    - 20 epocs -> 15,44% (995s)
    - 30 epocs -> 16,99% (1534s)

235M Params:
    on multi-GPU:
    - 10 epocs -> 19,39% (411s) 1er exec sous multi gpu
    - 20 epocs -> 11,37% (810s)
    - 30 epocs -> 16,73% (1220s)

