# Metal sources

These two files are Apple's documents, downloaded unchanged on 2026-10-20 from [Metal Resources](https://developer.apple.com/metal/resources/). They are not a rewrite and not a blog summary.

| File | What it is | This copy |
| --- | --- | --- |
| `Metal-Shading-Language-Specification.pdf` | The language: syntax, address spaces, attributes, the standard library, `simdgroup`, barriers. Apple's resources page calls it the definitive MSL specification. The same URL's text begins "Metal Shading Language Specification Version 4.1". | 383 pages, 14 MB |
| `Metal-Feature-Set-Tables.pdf` | Which GPU family has which feature and which limits. M4 is Apple9. | 18 pages, 2.9 MB |

Order when two sentences disagree:

1. A number this machine measured (`tcenv calibrate`, an evaluation report, or a fenced example in `msl_reference.md` that this compiler accepted).
2. These two PDFs.
3. An Apple WWDC session.
4. A blog. A blog is a lead. It does not go in as a fact.

`msl_reference.md` is the language note for this Metal backend. The specification file may describe a newer version. A feature is not available until this compiler accepts it. Chip peaks live in the target's `device.md`, not in this index.

## 两层，别混起来

| 文件 | 是什么 | 谁在用它 |
| --- | --- | --- |
| `Metal-Shading-Language-Specification.pdf`、`Metal-Feature-Set-Tables.pdf` | Apple 的**规范层**：语言怎么写、这一代芯片有什么 | 代理按 `SKILL.md` 给的名字去读 |
| `targets/<id>/experience.md` | 本仓库的**经验层**：我们在那块芯片上测到过什么 | 进系统提示，也拷进代理的工作区 |

`experience.md` 的每一条都带四行：`观察`、`来源`、`本机实测`、`边界`，外加 `Confidence:`。规则只有一条：

- `verified`：本机复现过，`来源` 里必须有 `runs/`、`docs/`、`tcenv/` 或 `taic/` 的路径。**只有这一级进提示词。**
- `reported`：只有上游出处（别人的 PR、博客、WWDC），还没在本机复现。留在库里，正文里写明「未验证」。

`tests/test_wiki_experience.py` 会检查这两条：缺 `Confidence`、`verified` 却没有来源或实测、`reported` 没有标未验证，都会让套件红。

### 怎么加一条 `reported`

从 mlx / metal-performance-shaders / mlx-examples / PyTorch MPS / Apple sample 近半年**已合并**的 PR 里找有性能说法的改动，抄「什么写法在什么条件下快」而不是抄代码：

```
## 一句话结论
Keywords: <会被问到的词>
Confidence: reported
观察: <写法 X 在条件 Y 下比 Z 快，上游怎么说>
来源: <repo>#<PR>（已合并）
未验证: 本机还没复现，别当事实用
边界: <dtype / shape / 系统版本>
```

之后要么在本机跑一次对照把它升成 `verified`，要么在 `experience.md` 之外保存，别让没测过的数字进上下文。

### 还没做的那一层

按已合并 PR 批量挖 know-how 的流水线（抓取、去重、人工过一遍、转成上面格式）**还没有**。现在的 `experience.md` 全是本仓库自己的实测；上游经验要靠人按上面的步骤加。
