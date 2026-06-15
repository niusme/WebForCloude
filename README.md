# WebAi — DeepSeek 网页版 × Claude Code 中转代理

> 🧸 **一个玩具性质的小项目** —— 用 Playwright 操控 [chat.deepseek.com](https://chat.deepseek.com) 的浏览器界面，把它包装成一个 Anthropic Messages API 兼容的 HTTP 服务。  
> 说白了就是：让 Claude Code 之类认 Anthropic API 的工具，能白嫖 DeepSeek 网页版的免费对话。

## 它做了什么？

DeepSeek 官方有网页版（免费），但没有公开 API。  
Anthropic 有完善的 API 生态（Claude Code、各种 SDK），但要付费。

**WebAi = 中间人**：用浏览器自动化把 DeepSeek 网页版的对话能力掏出来，再套上一层 Anthropic 兼容的 HTTP 接口。于是 Claude Code 就可以把 DeepSeek 当后端模型用了。

## 项目结构

```
src/
├── main.py           # CLI 入口：三种模式（login / chat / demo）
├── server.py         # FastAPI 服务：Anthropic Messages API 兼容
├── deepseek.py       # 核心客户端：Playwright 操控浏览器发消息、收回复
├── deepseek_api.py   # HTTP 直连方案：隐藏浏览器只算 PoW，聊天走 HTTP
├── browser.py        # 浏览器管理：启动/关闭、反检测脚本、会话持久化
├── config.py         # 集中配置：选择器、超时、反检测参数
├── prompt_builder.py # 提示词组装：Anthropic 消息格式 → DeepSeek 文本提示
├── tool_parser.py    # 工具调用解析：提取 <tool_call> XML 标签
├── exceptions.py     # 异常体系
└── timing.py         # 计时小工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
playwright install chromium
```

### 2. 登录（一次性的）

打开有头浏览器，手动扫码/邮箱登录，会话自动保存：

```bash
python src/main.py --mode login
```

登录成功后 `session/storage_state.json` 会保存登录态，后续操作无需重复登录。

### 3. 发条消息试试

```bash
python src/main.py --mode chat --message "用 Python 写个快排序"
```

### 4. 多轮对话演示

```bash
python src/main.py --mode demo
```

## 重点：Claude Code 中转代理

这是这个项目最有意思的用法 —— 把 DeepSeek 网页版包装成 Claude Code 的后端模型。

```bash
python src/server.py --port 8080
```

服务启动后会监听 `http://127.0.0.1:8080`，提供两个端点：

| 端点 | 说明 |
|------|------|
| `GET /v1/models` | 返回模型列表 |
| `POST /v1/messages` | 兼容 Anthropic Messages API |

然后在 Claude Code 配置里指向它：

```json
{
  "model": "deepseek-chat",
  "proxy": "http://127.0.0.1:8080"
}
```

所有 Claude Code 的 API 调用就会被转发到 DeepSeek 网页版后端 —— 包括工具调用（Bash、文件读写等）。

### 工具调用是怎么工作的？

这是一个有趣的工程挑战：

1. `prompt_builder.py` 把 Anthropic 的 tool 定义转为文本描述，塞进发给 DeepSeek 的提示词
2. DeepSeek 的回复会包含 `<tool_call>{"name": "Bash", "arguments": {...}}</tool_call>` 格式的 XML 标签
3. `tool_parser.py` 解析这些标签，转回 Anthropic 的 `tool_use` 格式
4. 工具执行结果通过 `--- TOOL EXECUTED ---` 标记回传给 DeepSeek

## 两种技术路线

这个项目里有**两套**实现方案，都在 `server.py` 里：

### 路线 A：全浏览器自动化（`deepseek.py`）

- 用 Playwright 打开真实 Chromium 浏览器
- 在文本框中打字、点发送、等回复
- 优点：行为完全模拟真实用户
- 缺点：慢、耗资源，需要等页面渲染

### 路线 B：HTTP 直连（`deepseek_api.py`）

- 仍然需要一个**隐藏的 headless 浏览器**，但只用来算 PoW（DeepSeek 的反爬验证）
- 真正的聊天请求直接发 HTTP 到 DeepSeek 的内部 API
- 优点：快很多，资源消耗小
- 缺点：依赖 DeepSeek 内部 API 的稳定性

`server.py` 默认走路线 B，PoW 浏览器在后台线程里常驻，每次请求前触发一次 PoW 计算，然后通过 HTTP 发消息。

## 反检测做了什么？

DeepSeek 对自动化访问有一定防御，`browser.py` 里做了一些对抗：

- 隐藏 `navigator.webdriver` 标志
- 填充 `navigator.plugins`（空数组是自动化的典型特征）
- 设置真实的语言和时区
- 禁用 `AutomationControlled` 特性
- 注入 `localStorage` 默认开启思维链（thinking）和搜索

这些都是猫鼠游戏，随时可能失效。

## 已知限制

- **PoW 验证**：DeepSeek 每次对话都需要 PoW，当前方案靠隐藏浏览器自动计算，如果 DeepSeek 升级验证方案就会失效
- **会话有效期**：登录态可能过期，需要重新 `--mode login`
- **工具调用不稳定**：DeepSeek 不是为 tool calling 设计的，生成的 JSON 有时格式不对，`tool_parser.py` 做了容错但不保证 100%
- **玩具性质**：代码里混杂着中文注释和日志，架构也没那么严谨 —— 这是探索性质的个人项目

## License

MIT — 随便玩。
