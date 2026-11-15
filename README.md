# 实验一：植物种类与生长状态多任务识别

基于 ResNet18 共享主干的**双头多任务**模型，一次前向同时输出植物**物种**与**生长状态**两类标签。对应《深度学习图像分析实验报告》第 2 章，代码不提交 BB 平台，统一托管于本仓库以便复查。

## 一、任务定义

| 任务 | 标签空间 | 说明 |
|---|---|---|
| 任务 1 | 物种：番茄、辣椒、生菜、黄瓜、小麦 | 5 类 |
| 任务 2 | 状态：健康、真菌病害、虫害、缺氮、干旱 | 5 类 |

两个标签空间相互独立，不做组合标签（避免类别数乘性膨胀与长尾）。

## 二、模型结构

```
输入 224×224×3
   └─ ResNet18（ImageNet 预训练，fc 替换为 Identity）→ 512 维特征
        ├─ Dropout(0.3) → Linear(512→5)  # 物种头
        └─ Dropout(0.3) → Linear(512→5)  # 状态头
```

损失函数：

```
L_total = α * CE(species) + (1-α) * CE(status)      α 默认 0.5
```

## 三、环境依赖

```
python >= 3.8
torch >= 2.0
torchvision >= 0.15
numpy
pillow
scikit-learn
tqdm
```

```bash
pip install -r requirements.txt
```

## 四、数据准备

数据不入库，只保留划分清单。需准备 `data/train.csv`、`data/val.csv`、`data/test.csv` 三份文件，每行格式：

```csv
image_path,species,status
images/tomato_0001.jpg,tomato,healthy
images/tomato_0002.jpg,tomato,nitrogen_deficiency
```

仓库内附 `data/train_sample.csv` 作为格式示例。类别名称与索引的对应关系在 `configs/labels.json` 中维护，训练脚本按该文件顺序编码。

划分要求：训练 70%、验证 20%、测试 10%，**按采集批次与植株分组**，同一植株的图像不得跨集合。

## 五、训练

```bash
python train.py --train_csv data/train.csv --val_csv data/val.csv --epochs 40 --alpha 0.5
```

训练策略（两阶段迁移）：

1. 阶段一：冻结主干，只训练两个分类头，lr = 3e-4，10 轮；
2. 阶段二：解冻全部参数整体微调，lr = 1e-5，30 轮，余弦退火；
3. 每轮记录两个分支准确率，以均值作为模型选择依据，保存 `best_plant_model.pth`。

## 六、推理

```bash
python inference.py --image test.jpg --weights best_plant_model.pth
```

输出示例：

```
预测植物种类：番茄，置信度：0.968
预测生长状态：缺氮，置信度：0.893
```

## 七、参考结果（课程条件下的参考量级）

| 分支 | 准确率/% | 宏平均 F1/% |
|---|---|---|
| 物种分类 | 96.8 | 96.4 |
| 状态分类 | 89.3 | 88.2 |

消融（主干均为 ResNet18）：

| 配置 | 物种 | 状态 | 平均 |
|---|---|---|---|
| 独立双模型 | 96.2 | 87.4 | 91.8 |
| 多任务 α=0.3 | 96.5 | 88.9 | 92.7 |
| 多任务 α=0.5 | 96.8 | 89.3 | 93.1 |
| 多任务 α=0.7 | 97.1 | 88.1 | 92.6 |

> 上表用于说明指标口径与对比关系，实际数值以本机运行日志为准。

## 八、注意事项

- 增强禁区：不得使用大幅度色相偏移、通道交换、垂直翻转。颜色本身就是缺素与病害的判据，色相扰动会把标签信息破坏掉。
- 状态类别极度不均衡时，建议对 `CE(status)` 加类别权重（本仓库已默认启用），或改用 Focal Loss。
- 若要替换主干（ResNet50 / EfficientNet-B0），只需改 `--backbone` 参数，输出维度会自动适配两个头。

## 九、目录

```
.
├── README.md
├── requirements.txt
├── configs/labels.json
├── data/                 # 仅放划分清单 csv，不放原图
├── model.py
├── train.py
├── inference.py
```
