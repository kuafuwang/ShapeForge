# `python -m observe` 的入口，只调用 observe.app.main。路由和 HTML 都在那一次导入里固定。
# 页面上的数来自索引。改 app、索引或 static/index.html 之后要重新执行这条命令。浏览器刷新拿到的仍是旧进程；runs/ 里的新结果下一帧就能看见。

from observe.app import main

# 被别的模块 import 时不要绑端口。只有以脚本方式运行才启动服务。
if __name__ == "__main__":
    main()
