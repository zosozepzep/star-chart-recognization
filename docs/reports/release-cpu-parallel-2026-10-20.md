# CPU 并行版交付与目录整理

版本：`national-cpu-parallel-20261020`。源码与完整离线演示包位于 `dist/cpu-parallel-20261020/`。
算法、性能及全量回归结果见 [CPU 并行实测](cpu-parallel-2026-10-20.md)。

## 交付内容

- `star-chart-source-national-cpu-parallel-20261020.zip`：当前源码、配置、测试、文档与离线 Gaia 星表；按 200,000,000 字节上限检查。
- `star-chart-offline-national-cpu-parallel-20261020.zip`：包含 `offline-kit/`，内有程序、15 张官方 FITS、升级后的 CPU 镜像、并行对照结果及科学验证材料；解压后运行 `start.ps1`。
- 两个 ZIP 各自的 `.sha256`、`build-receipt.json` 和 `release-receipt.json`。
- 可直接运行的 `offline-kit/`，方便本机演示；其文件应与完整 ZIP 相同。

完整包的 `KIT_MANIFEST.json` 逐文件记录大小与 SHA-256。应用的 `SOURCE_MANIFEST.json` 固定源码，
`RELEASE.json` 固定输入数据和镜像 ID。生成清单后校验 ZIP 中每个文件；从完整 ZIP 解压到新的临时目录，
执行源码 / 数据 / 镜像核验，并在禁网容器中用打包代码跑官方 15 帧，与已验证 CSV 比较。
完整包验收回执记录 Git 提交、精确文件大小与校验值、解压复验结果；避免把包自身校验值写入包内造成循环依赖。

源码和完整离线包属于本地交付物，不上传至 Git 仓库。Git 推送包含实现、测试及文档，交付物由上述提交打包。

## 清理边界

新包验证完成后，删除已被压缩包完整覆盖的上一版解压目录、早期重复源码 ZIP、临时源码副本、浏览器配置及工具依赖缓存。
上一版完整 ZIP 与源码 ZIP 保留一份用于回退，其余历史回执按原路径归档。
原始 FITS、正式分析输出、注入与拖尾实验、PPT/视频、历史设计文档和既有研究归档均保留。

清理使用明确路径清单；递归删除前确认路径位于项目内部且没有重解析点。清单保存每个文件的原始 SHA-256，
对可替代的解压副本先比较 ZIP 内容，删除后重新核验当前源码与 15 张原始 FITS。记录位于 `archive/cleanup-20261020/`。
