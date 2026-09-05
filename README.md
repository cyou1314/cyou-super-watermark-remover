# cyou · 超级去水印

本地优先的短视频固定水印清理工具。项目当前处于公开技术预研阶段，尚未发布可用安装包。

> 当前目标不是“万能一键无痕”，而是在明确边界内，把15秒以内短视频的固定位置可见水印处理得稳定、可预览、可复核。

## 当前范围

- 第一批用户：项目作者本人。
- 首发平台：Windows。
- 视频范围：通常约10秒，最长不超过15秒。
- 第一阶段：固定位置可见水印；MP4作为首个基准格式。
- 处理方式：默认本地运行，原始素材不上传。
- 验证设备：Intel Core i7-14700F、32GB内存、NVIDIA GeForce RTX 5060 8GB。

## 计划中的基本流程

```text
导入视频
  → 检查格式、分辨率、帧率、时长和音频
  → 手动画框标记固定水印
  → 生成5秒局部预览
  → 调整区域或处理模式
  → 分段处理完整视频
  → 检查残边、闪烁、音画参数和文件完整性
  → 保留原音频并另存导出
```

## 明确边界

- 仅用于处理用户本人拥有或已经获得授权的素材。
- 不提供粘贴第三方平台链接、批量下载并移除平台标识的功能。
- 完全不透明的水印会覆盖原始像素，修复结果只能是推断或重绘，不等于恢复原始内容。
- 第一阶段不支持移动水印、满屏平铺水印、大面积中央遮挡和必须精确还原的人脸、文字、二维码或商品细节。
- 技术导出成功不代表画面质量已经通过，结果仍需要视觉复核。

完整说明见[产品边界](docs/PRODUCT_BOUNDARIES.md)。

## 当前路线

- [x] 确认产品、仓库、开源和首发范围。
- [ ] 建立有无水印原片对照的固定视频测试集。
- [ ] 对比传统插值、单帧修复和时序修复路线。
- [ ] 确认第一版可商用、可分发的依赖。
- [ ] 完成命令行技术原型。
- [ ] 完成Windows本地网页式MVP。
- [ ] 小范围自用和真实创作者测试。
- [ ] 发布1.0开源版。
- [ ] 根据采用率、维护成本和模型许可决定付费入口。

详细阶段和通过条件见[路线图](docs/ROADMAP.md)，首轮私人样本的公开元数据见[测试基线](docs/TEST_BASELINE.md)。

## 开源与未来商业路线

社区版采用[GNU Affero General Public License v3.0](LICENSE)。项目会先公开验证和迭代；产品成熟后，可能对官方签名构建、自动更新、高级批量工作流、获得合法商用许可的专业模型、技术支持或云端GPU处理收费。

已经公开的社区核心不会通过改名被假装成全新闭源能力。未来如提供商业授权，将保持许可证和模块边界清晰。

在贡献者授权规则完成前，暂不接收代码Pull Request；欢迎使用Issue提交复现步骤、建议和不含私人素材的测试信息。参见[贡献说明](CONTRIBUTING.md)。

## 隐私与素材

- 不要在Issue、Discussion或Pull Request中上传没有公开授权的视频、图片或个人信息。
- 大型模型权重、用户素材、处理缓存、密钥和本地登录状态不会进入仓库。
- 如未来收集失败样本，必须单独取得素材所有者明确授权。

## 当前状态

`v0.0.x-research`：只建立可复现的测试与技术结论，不承诺生产可用性。

前三条固定矩形基线已经跑通。FFmpeg `delogo`、OpenCV Telea和OpenCV Navier–Stokes都能快速清除文字，但复杂背景均会形成明显涂抹矩形，因此只作为后续方案必须超过的速度/质量下限，不作为产品处理引擎。

研究脚本示例：

```powershell
.\scripts\run-delogo-baseline.ps1 `
  -InputPath 'D:\media\input.mp4' `
  -OutputPath 'D:\media\output-delogo.mp4' `
  -X 100 -Y 100 -Width 120 -Height 40
```

脚本拒绝覆盖原文件或已有输出，使用H.264高质量编码并直接复制原音频。运行前需自行安装带`delogo`和`libx264`的FFmpeg构建；实际分发许可必须按构建选项重新审计。

OpenCV研究基线：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe .\scripts\run-opencv-inpaint-baseline.py `
  --input 'D:\media\input.mp4' `
  --output 'D:\media\output-telea.mp4' `
  --x 100 --y 100 --width 120 --height 40 `
  --method telea --radius 3
```

OpenCV脚本只修复蒙版附近的小裁剪区，再把结果写回原帧；这样可以保留基线算法效果并减少无意义的整帧计算。`--method`支持`telea`和`ns`。

## 名称

- 产品名：`cyou`
- 中文项目名：超级去水印
- GitHub仓库：`cyou-super-watermark-remover`
