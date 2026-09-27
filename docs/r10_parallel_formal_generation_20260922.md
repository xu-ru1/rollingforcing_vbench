# R10：六卡正式生成 wave

R10 将六个冻结方法固定分配为：Vanilla→GPU 2、Fixed-slow→3、Front-slow→4、
Fixed-fast→5、Front-fast→6、U-shape-fast→7。一个 wave 的所有方法生成同一段连续的
unique-prompt index；同一 prompt 的六种策略并行产生，而每种策略只由一张卡串行执行。

wave 的输出通过 R9 的 raw commit、480-frame pixel-exact crop 与 VBench 标准命名校验后，
才写入各自 `status.json`。失败的 worker 不影响其他已成功方法；重试前保留日志并仅处理失败
分片。R10 不执行 VBench metric。
