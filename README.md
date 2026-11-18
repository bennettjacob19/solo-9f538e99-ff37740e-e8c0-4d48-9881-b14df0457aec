# 容错分布式逻辑回归训练台（单任务）

质检缺陷分类器的训练网页。两个**独立 CPU 进程**各自持有互斥的样本分片，每轮
交换结果、合并为**全数据平均梯度**后**同步更新**；任一进程退出即暂停整组，
容器重启后可从最后已确认轮恢复，恢复结果与不中断训练**逐位一致**。

## 1. 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000
```

三个容器、三个独立持久卷：

| 服务 | 角色 | 持久卷 |
| --- | --- | --- |
| `coordinator` | FastAPI 网页 + 编排 + 确认落盘 + 权重工件 | `coord_data`（`/data`）|
| `worker1` | 独立 CPU 训练进程，持有前 ⌈n/2⌉ 行 | `worker1_data`（`/data`）|
| `worker2` | 独立 CPU 训练进程，持有其余行 | `worker2_data`（`/data`）|

停止后数据保留：`docker compose down` 再 `up` 仍可恢复；如需清空用
`docker compose down -v`。

## 2. 数据与训练规格

* CSV：两列数值特征 `x1,x2` + 一列 0/1 标签 `label`，2～200 行，可含表头。
* 逻辑回归：增广向量 `[1, x1, x2]`，权重含偏置共 3 个，float64。
* 损失：全数据二元交叉熵（BCE）；参数 **w 与动量 v 均零初始化**。
* 超参固定：学习率 `lr=0.1`、动量 `momentum=0.9`；轮数 1～100。
* 更新（经典动量 SGD）：`v ← 0.9·v − 0.1·grad`，`w ← w + v`。

## 3. 网页操作

1. 选择 CSV、填写轮数 → **上传并建组**（自动做确定性互斥分片并下发）。
2. **开始**：打开一个训练代（generation），两侧对齐到已确认轮后逐轮训练。
3. 页面显示：已确认轮次、最新全数据损失、损失曲线、两个进程的 PID/代/持久化轮次。
4. 故障注入：
   * **人工延迟**：对每个进程的 `prepare/gradient/commit` 响应加 0～120 秒延迟；
   * **强杀 worker1/worker2（SIGKILL）**：容器入口守护自动重启该进程。
5. 进程退出后整组自动 **paused**；点 **故障后恢复（Resume）** 继续。
6. **下载已确认权重**：只对应最后已确认轮（`weights_latest.json`）。

## 4. 容错与一致性协议（为什么恢复后数值一致）

### 4.1 轮协议
每轮协调者并行向两个 worker 发 `gradient`，worker 返回各自分片的**逐样本残差**
`p−y`；协调者按数据集原始行序拼接残差，用**一次**矩阵乘法重建全数据平均梯度
（与单进程参考实现完全相同的求和顺序），再把同一梯度并行发给两个 worker 做
`commit`。worker 在返回前用 **temp+fsync+原子 rename** 落盘新的 `w/v/round`。

### 4.2 确认规则（不跳轮、不重复应用）
* 只有**两侧同一轮、同代的参数与动量都完整持久化**后，协调者才原子写入
  `status.json` 确认该轮；确认前页面与权重都不会前进。
* 允许重做未确认轮；worker 的 `commit` 对已持久化轮幂等（返回 `dedup=true`，
  不二次应用），且拒绝非 `当前轮+1` 的提交，因此无法跳轮。
* 协调者自身在轮中被杀：重启 lifespan 把 `running` 改为 `paused`，未确认轮作废，
  下次 Resume 经 `/prepare` 把两侧回滚到最后确认轮后重做。

### 4.3 代际围栏（旧组迟到消息不能推进新组）
每条协议消息都带 `task_id + gen`。每次开始/恢复 `gen` 严格 +1，worker 只服务
`/prepare` 安装的当前代；旧代或未知任务的消息一律 `409/410` 拒绝，不改变任何状态。

### 4.4 持久化与权重
* 所有 JSON 均原子写（临时文件 → fsync → `os.replace` → 目录 fsync），读者只会
  看到上一完整版本或新版本，不会看到半写文件。
* 权重每轮先写不可变文件 `weights_round_XXXX.json`，最后原子切换 `weights_latest.json`；
  保存失败时上一完整轮工件保持不变，下载内容永远只对应已确认轮。

## 5. 故障演练

```bash
# 训练中强杀一个进程（网页按钮等价）
docker compose kill worker1
# 观察网页：整组 paused；worker1 入口守护自动重启
# 点“故障后恢复”，最终损失/权重与不中断训练一致

# 协调者（网页自身）在轮中重启
docker compose restart coordinator

# 全集群重启（卷保留）
docker compose down && docker compose up -d
```

内置参考轨迹接口 `GET /api/reference` 返回同输入单进程不中断训练的逐轮
权重/动量/损失，测试用它做逐位比对。

## 6. 本地测试（无需 Docker）

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -q
```

* `tests/test_lrmath.py`：分片残差重建梯度与单进程梯度逐位相等、可重复、数值稳定。
* `tests/test_integration.py`：拉起 3 个真实 uvicorn 进程，覆盖
  干净训练一致、worker 强杀恢复、协调者轮中重启、全集群重启、旧代消息围栏、
  commit 幂等去重、`/prepare` 回滚、CSV 校验、仅确认轮可下载权重。

## 7. 目录结构

```
app/
  lrmath.py        # 纯 NumPy 逻辑回归数学（float64，无随机数）
  storage.py       # 原子/崩溃安全 JSON 持久化
  worker.py        # 独立训练进程：分片、围栏、幂等提交、强杀/延迟
  coordinator.py   # FastAPI 网页与编排：状态机、确认、权重工件
  web/index.html   # 单文件操作网页（无外部 CDN）
  entrypoint-*.sh  # 容器入口（worker 含 SIGKILL 后重生守护）
tests/             # 单元 + 端到端故障注入测试
docker-compose.yml # 三服务 + 三个独立持久卷 + 健康检查
Dockerfile
```
