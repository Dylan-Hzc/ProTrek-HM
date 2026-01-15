# 生命科学中的信息检索问题

本项目用于演示与实践生命科学中的信息检索方法，核心示例基于 BLAST 的序列相似性检索与结果解析。代码部分主要运行 Jupyter Notebook：

- 绝对路径（示例）：`C:\Users\PS\Desktop\25fall\生命科学中的信息检索问题\code\BLAST.ipynb`
- 相对路径（仓库内）：`code/BLAST.ipynb`

> 建议在本仓库根目录启动 Jupyter，并在界面中打开 `code/BLAST.ipynb` 逐单元运行或“Run All”。

## 目录结构

- `code/`：核心 Notebook 与相关脚本（`BLAST.ipynb`）
- `data/`（可选）：本地序列输入文件（例如 FASTA）
- `results/`（可选）：运行生成的中间或导出结果
- `README.md`：项目说明

## 准备环境

- 操作系统：Windows 10/11（macOS/Linux 同样适用，命令略有差异）
- 基础依赖：Python 3.9+，Jupyter（Notebook 或 JupyterLab）
- 建议工具：Conda 或 Python venv（二选一）

需要的 Python 包（最小集合）：

- jupyter 或 jupyterlab
- biopython
- pandas、numpy、matplotlib、seaborn（用于数据整理与可视化，按需）

## 依赖与可选组件

- Python 与 Jupyter：用于运行与展示 Notebook。
- Biopython：若 Notebook 使用 `Bio.Blast` 模块调用在线 BLAST 接口，需要稳定网络；具体实现见 Notebook 单元。
- 可选（本地 BLAST+）：如果 Notebook 配置为调用本地 BLAST，需要预先安装 NCBI BLAST+ 并将可执行文件加入系统 PATH（`blastn`/`blastp` 等）。如使用本地数据库，还需准备并索引相应数据库。

> 在线 BLAST 服务通常有频率限制；请避免高并发与过于频繁的请求。

## 数据与路径

- 若需要本地输入序列文件，请将 FASTA/文本数据放入 `data/`（或在 Notebook 中调整为你的文件路径）。
- 若因中文路径导致个别第三方库报错，可临时将仓库移动至仅包含英文字符的路径后重试。

