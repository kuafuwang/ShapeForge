# ShapeForge 文档可视化

这是一个离线文档阅读器，参考 `tx82_planner/docs/doc_visual` 的分类首页、阅读页和 Mermaid 展示，但数据由本仓库当前 Markdown 生成。

直接用浏览器打开 `index.html` 即可。也可以用系统静态服务器：

```bash
uv run python -m http.server --directory docs/visual 8000
```

然后打开 `http://127.0.0.1:8000/`。不需要安装 npm 依赖，不请求 CDN，不调用模型或 GPU。

## 页面

- `index.html`：分类、全文搜索、按角色的阅读路径；
- `reader.html?doc=<仓库相对路径>`：目录、正文、文档内查找、源文件链接、Mermaid；
- `assets/`：本项目的页面样式和交互；
- `data/`：由构建器生成的文档数据与指纹；
- `vendor/`：本地前端库，不联网加载。

## 修改文档后重建

```bash
uv run python scripts/build_doc_visual.py
uv run python scripts/build_doc_visual.py --check
```

构建器扫描：项目入口 Markdown、`docs/**/*.md`（排除本目录）、backend wiki、以及测试/后端/目标的说明文档。`docs/visual/README.md` 不进入索引，避免生成器把自己当输入。

`data/docs.js` 和 `data/manifest.json` 都是生成物，不要手改。`--check` 只比较内容是否一致，不写文件；`tests/shared/test_doc_visual.py` 用它防止漏重建。

## 离线和链接

- 阅读器用 `marked` 渲染 Markdown，用 `DOMPurify` 清理 HTML，用 `Mermaid` 渲染图；
- 文档正文内嵌在 `data/docs.js`，不使用 `fetch()`，因此支持 `file://`；
- 指向其它 Markdown 的链接会重写为 `reader.html?doc=...`；
- 指向 PDF、图片、JSON、源码的链接按源文档所在目录解析；
- 外部 HTTP 链接仍指向原地址，点击时需要网络，但页面本身不依赖它们。

## 第三方库

- `vendor/marked.umd.js`：marked，MIT；
- `vendor/purify.min.js`：DOMPurify，Apache-2.0 / MPL-2.0；
- `vendor/mermaid.min.js`：Mermaid，MIT。

对应许可证在 `vendor/LICENSE-*.txt`。升级时替换同名库文件和许可证，再运行页面测试。仓库没有 npm build step。
