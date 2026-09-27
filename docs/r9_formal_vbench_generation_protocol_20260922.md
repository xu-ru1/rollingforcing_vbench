# R9：正式 VBench 生成分片

每个 R9 分片直接生成 R8 冻结计划中的正式样本，不设置独立的 GPU smoke。分片使用冻结的
effective config、checkpoint、126 latent、seed 0 与每 prompt RNG reset。`start` 和 `count`
在 944 个唯一 prompt 上索引；六方法总计 5664 个原始和 5664 个 480 帧评测视频。

生成入口保持 R7 已验证的 indexed 文件名。分片完成后，raw video 被逐字节提交为全局索引名，
经固定 crop 后再逐字节硬链接或复制为 VBench standard mode 所需的完整 prompt 文件名。每一步
均写 SHA256、crop pixel-exact 与 audit 状态；仅写出 `status.json: ok` 的分片可计入正式数据。

`--plan-only` 不使用 GPU；`--execute` 才生成该分片。已完成分片再次执行会直接退出，防止无意
重复生成。VBench metric 与 temporal-flickering static filter 留到所有正式视频完整后运行。
