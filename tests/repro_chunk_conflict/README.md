# Btrfs chunk 映射来源冲突诊断

本目录回答一个取证问题：当超级块 `sys_chunk_array` 与磁盘上的 chunk tree
对同一段逻辑地址给出的映射不一致时，`dissect.btrfs` 当前到底采用哪一份？
如何区分**无害的重复条目**与**会改变证据来源的冲突**？

## 结论速览

当前读取器采用“先入者胜（first-writer-wins）”，且判定依据**只有逻辑覆盖关系，
从不比较条目内容**。

- 加载顺序固定：`Btrfs._initialize_chunks()` 先解析超级块 `sys_chunk_array`，
  再遍历 chunk tree（见 `dissect/btrfs/btrfs.py:131`）。
- 两者都调用 `ChunkStream.add(offset, chunk)`。该方法在
  `dissect/btrfs/stream.py:78` 做一次 `bisect_right` 前驱查找，只要新条目的
  起点落在某个已存在 chunk 的 `[offset, offset+length)` 区间内，就**直接
  `return` 丢弃新条目**，不比较长度、类型、stripe、设备 UUID（`stream.py:79-83`）。
- 因为超级块条目先加入，所以只要 chunk tree 条目的起点被超级块条目覆盖，
  **胜出的永远是超级块 `sys_chunk_array` 那份**，无论它与 chunk tree 差异多大。
- 唯一例外：若新条目的起点位于“间隙”（没有任何已存在 chunk 覆盖该点），
  前驱检查为空，条目会被插入；此时它可能与一个**起点更晚**的 chunk 重叠，
  而 `add()` 完全不检测这种重叠（见场景 E）。

读取阶段同样按逻辑偏移用 `bisect_right` 选前驱 chunk（`stream.py:121`、`137`），
因此最终逻辑字节来自“覆盖该地址且起点最高的已加载 chunk”。`stripe.dev_uuid`
会被解析进 `Stripe`，但整条读取链路从不校验它与实际挂载设备是否一致
（设备只按 `devid` 索引，见 `btrfs.py:71`、`stream.py:91`）。

## 为什么“逻辑起点相同就跳过”不够

起点相同存在两种本质不同的情况，当前代码把它们一视同仁地丢弃：

1. **完全相同的重复条目**：两份描述解析到完全相同的物理位置
   （同样的 devid、物理偏移、设备 UUID、类型与条带几何）。丢弃任一份都不改变
   字节来源，属于安全去重——这正是健康文件系统的常态（chunk tree 是
   `sys_chunk_array` 引导信息的超集，引导 chunk 在两处字节级一致）。
2. **同起点、异内容的冲突**：逻辑起点相同，但 stripe 指向另一设备/另一物理偏移，
   或设备 UUID 不同。两份描述会把同一逻辑地址映射到不同物理字节，选取哪一份
   直接改变证据来源。当前实现静默保留先到的一份，不发出任何告警。

因此去重判据必须从“起点是否被覆盖”升级为“覆盖条目与候选条目是否**指纹相同**”。

## 冲突判定建议

判定在 `chunk_conflict.py` 中实现，对读取器真实生成的 `Chunk` 命名元组做
**规范化指纹比较**（逻辑跨度、profile/类型、条带几何、每条 stripe 的
`(devid, 物理偏移, 设备 UUID)`）。本模块**不自动选择“较新”一份**，只给处置动作。

| 情况 | 判定 kind | 建议动作 | 理由 |
| --- | --- | --- | --- |
| 起点相同且整条指纹一致 | `duplicate` | `deduplicate` 安全去重 | 两份解析到同一物理字节，丢弃不改变证据 |
| 起点相同但 stripe / UUID / 几何 / 类型不同 | `hard_conflict` | `block_extraction` 阻止提取 | 同一逻辑地址有两个物理来源，字节取决于谁胜出 |
| 起点不同但逻辑区间部分重叠 | `overlap_conflict` | `block_extraction` 阻止提取 | 重叠子区间有两个属主，没有唯一映射 |
| 起点在间隙、却与更晚 chunk 重叠（场景 E） | `overlap_conflict` | `block_extraction` 阻止提取 | 读取器会插入并造成并存的双重属主 |
| 引用的设备既未挂载又超出 RAID 容错 | 读取器直接抛 `Error` | 阻止并报告缺失设备 | 该映射根本不可读，禁止猜测 |
| 与任何已加载 chunk 不重叠的新区间 | `new_range` | `accept` | 正常的 chunk tree 扩展（数据 chunk 本就只在树里） |

要点：

- **可以安全去重的只有指纹完全一致的重复条目**（场景 A）。
- **同起点异条带、部分重叠、设备 UUID 不同都应阻止文件内容提取**（场景 B/C/D/E），
  因为它们改变证据来源；正确做法是报告冲突、保留两份原始描述，交给分析人员裁决，
  而不是依据世代号或加载顺序静默选一份。
- 不应自动选“较新”：超级块世代与 chunk tree 条目的更新不是同一事务的可靠先后证据，
  且损坏/篡改场景下“较新”可能正是被植入的映射。

## 复现实验（可运行）

复现直接驱动真实的 `ChunkStream.add/_read`，与生产代码路径一致；按真实加载顺序
先喂 `SB`（超级块）再喂 `CT`（chunk tree），每个物理区域写入不同单字节标记，
从而能明确指出每个逻辑字节实际来自哪里。

```bash
# 直接运行，打印每个场景的判定、读取器真实选择与字节来源
.venv/bin/python -m tests.repro_chunk_conflict.reproduce_conflicts

# 作为 pytest 运行（含断言与缺失设备场景）
.venv/bin/python -m pytest tests/repro_chunk_conflict/ -q
```

### 对照矩阵与实测结果

| 场景 | 构造 | 判定 | 读取器实际行为 | 逻辑字节实测来源 |
| --- | --- | --- | --- | --- |
| A 完全一致 | 同起点、同 stripe/UUID | duplicate / 去重 | 保留 SB，丢弃 CT | SB/CT 共享区域（两者本就同一位置） |
| B 同起点异条带 | CT 指向另一 devid+偏移 | hard_conflict / 阻止 | 保留 SB，丢弃 CT | SB 区域（devid=1, phys=0x0） |
| C 部分重叠 | CT 起点落在 SB 区间内 32 KiB 处 | overlap_conflict / 阻止 | 保留 SB，丢弃 CT | SB 区域（重叠子区间的属主仍是 SB） |
| D 不同设备 UUID | 同 devid/偏移、UUID 不同 | hard_conflict / 阻止 | 保留 SB，丢弃 CT，且从不校验 UUID | SB 区域（UUID 差异被完全忽略） |
| E 间隙起点重叠 | CT 起点在间隙但与更晚 SB chunk 重叠 | overlap_conflict / 阻止 | **插入 CT**（无前驱覆盖），但读取仍由 SB chunk 胜出 | SB 区域；列表里同时存在两个属主 |

B/C/D 中 `chunks loaded` 始终为 1，证明冲突条目被静默丢弃；E 中为 2，
证明该路径会插入并留下双重属主。所有阻止类场景在判定层都被标记为
`block_extraction`，与读取器“静默选一份”的现状形成对照。

## 建议的落地方式（不改变现有读取结果）

`chunk_conflict.classify_candidate()` 是纯函数、无副作用，可在
`Btrfs._initialize_chunks()` 的每次 `add` 旁挂接：先分类、记录指纹与来源
（SB/CT），再决定是否继续调用现有 `add`。遇到 `block_extraction` 时抛出
专用异常或在文件系统对象上记录冲突清单，使后续 `INode.open()` 拒绝返回内容，
而不是静默返回先到映射对应的字节。这样既保留现状用于兼容，又能在取证模式下
保证证据来源不被静默替换。

## 文件清单

- `chunk_conflict.py`：指纹、`classify_candidate()` 判定、`explain_reader_choice()`
  对当前读取器行为的只读解释。
- `reproduce_conflicts.py`：五个对照场景的可运行复现，输出每个逻辑字节的真实来源。
- `test_chunk_conflict.py`：对判定结论、读取器真实选择与缺失设备异常的 pytest 断言。
