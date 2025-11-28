# Hello Agents

一个 AI Agent 学习项目，分为三个章节，逐步学习如何构建智能 Agent。

## 项目说明

这个项目包含三个独立的学习章节：

**Chapter 1**: 基础 Agent - 学习如何用 LLM 调用工具
- 实现一个旅行助手 Agent
- 使用天气查询和景点推荐工具
- 理解 Thought-Action-Observation 循环

**Chapter 2**: 本地模型 - 使用本地模型进行推理
- 使用 Qwen 模型本地推理
- 无需调用 API，支持 GPU 加速

**Chapter 3**: ReAct 框架 - 更强大的推理能力
- 改进的工具执行器
- 更好的推理链支持

## 目录结构

```
Hello_agents/
├── Chapter1/               # 基础 Agent
│   ├── codes/
│   │   ├── main.py         # 旅行助手示例
│   │   ├── llm.py          # LLM 类
│   │   └── tools/          # 工具集
│   └── image/
├── Chapter2/               # 本地模型
│   └── llm_call.py
└── Chapter3/               # ReAct 框架
    ├── ReAct.py
    └── tool/
```

## 安装和使用

### 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n hello_agents python=3.10

# 激活环境
conda activate hello_agents
```

### 安装依赖

```bash
# Chapter 1 & 3 依赖
pip install openai python-dotenv

# Chapter 2 依赖（PyTorch 和 Transformers）
pip install torch torchvision torchaudio transformers

# 或者使用 conda 安装 PyTorch（推荐 GPU 用户）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 配置（Chapter 1 & 3）

**第一次使用时**：
```bash
cp .env.example .env
```

然后编辑 `.env` 文件，填入你的 API 密钥：
```env
OPENAI_API_KEY=your_openai_key
OPENAI_API_BASE_URL=https://api.openai.com/v1
TAVILY_API_KEY=your_tavily_key
SEARCH_API_KEY=your_search_key
```

**重要**：`.env` 文件已被加入 `.gitignore`，你的 API 密钥不会被提交到仓库

### 运行
```bash
# Chapter 1
cd Chapter1/codes
python main.py

# Chapter 2
cd Chapter2
python llm_call.py

# Chapter 3
cd Chapter3
python ReAct.py
```

## 工作流程

Agent 的基本流程：
1. 用户提问
2. LLM 思考并决定使用哪个工具
3. 执行工具，获得结果
4. 反馈给 LLM，继续推理
5. 重复直到得出最终答案

```
用户输入 → LLM 思考 → 选择工具 → 执行 → 观察结果 → 迭代 → 最终答案
```

## 常见问题

**Q: 导入错误？**
A: 确保使用相对导入。例如在 `tool/` 目录下应该使用：
```python
from .search_tool import search
```

**Q: API 密钥错误？**
A: 检查 `.env` 文件是否正确配置

**Q: 本地模型很慢？**
A: 使用更小的模型，或启用 GPU 加速

**Q: 我不小心提交了 API 密钥怎么办？**
A: 立即撤销提交并删除密钥：
```bash
# 查看提交历史
git log --oneline

# 重置到提交前
git reset --soft HEAD~1

# 编辑 .env 和 .gitignore
git add .gitignore .env.example
git commit -m "Remove .env from tracking"
git push --force
```

## 许可证

MIT License
