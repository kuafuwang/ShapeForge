"""The dashboard sends an ETag and gzip, and answers If-None-Match with 304."""

from starlette.testclient import TestClient

from observe.app import app


def test_the_run_similar_endpoint_serves_one_run_and_404s_for_an_unknown_run(monkeypatch):
    """@brief 按需相似结论端点的路由：存在的 run 返回 items，不存在的返回 404。

    @param monkeypatch 用假的 query.feedback_similar 隔离索引与句向量，只验路由与状态码。
    """
    from observe import query

    def fake(run_id):
        return {"run_id": run_id, "items": []} if run_id == "a/one" else None

    monkeypatch.setattr(query, "feedback_similar", fake)
    with TestClient(app) as client:
        ok = client.get("/api/v2/runs/a/one/similar")
        assert ok.status_code == 200 and ok.json() == {"run_id": "a/one", "items": []}
        assert client.get("/api/v2/runs/a/none/similar").status_code == 404


def test_the_page_and_json_carry_an_etag_and_revalidate(tmp_path, monkeypatch):
    """@brief 页面返回 gzip 压缩和 ETag，If-None-Match 命中时返回 304；JSON 接口也带 ETag。

    @param tmp_path pytest 提供的临时目录，用作 runs、数据库和 books 路径。
    @param monkeypatch pytest 补丁工具，用于设置三个路径环境变量。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        etag = page.headers["etag"]
        assert page.headers.get("content-encoding") == "gzip"
        again = client.get("/", headers={"If-None-Match": etag})
        assert again.status_code == 304 and again.content == b""

        rows = client.get("/api/v2/runs")
        assert rows.status_code == 200 and rows.headers.get("etag")


def test_the_analysis_endpoints_require_an_epoch_and_honour_stamps(tmp_path, monkeypatch):
    """@brief 三个分析端点：混纪元 400（带可选纪元）、gap 的 stamp 短路、committees/noise 可读。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    runs = tmp_path / "runs"
    for name, protocol in (("a/one", "3"), ("b/two", "flame_chase_1")):
        directory = runs / name
        directory.mkdir(parents=True)
        (directory / "state.json").write_text('{"task": "relu", "dtype": "fp16"}')
        (directory / "result.json").write_text(
            '{"task": "relu", "dtype": "fp16", "stop_reason": "max_rounds", "protocol_version": "%s"}' % protocol)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(runs))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    from observe.index import Index

    Index(tmp_path / "observe.sqlite").sync_known()
    with TestClient(app) as client:
        mixed = client.get("/api/v2/gap")
        assert mixed.status_code == 400
        assert mixed.json()["epochs"] == ["2", "3"]
        gap = client.get("/api/v2/gap?protocol_epoch=3")
        assert gap.status_code == 200
        stamp = gap.json()["stamp"]
        assert client.get(f"/api/v2/gap?protocol_epoch=3&stamp={stamp}").json() == {
            "same": True, "stamp": stamp,
        }
        assert client.get("/api/v2/committees").status_code == 400
        assert client.get("/api/v2/committees?protocol_epoch=3").status_code == 200
        noise = client.get("/api/v2/noise?protocol_epoch=3")
        assert noise.status_code == 200 and noise.json()["cache_hit"]["total"] == 0
        assert client.get("/api/v2/noise?protocol_epoch=3&stamp=" + noise.json()["stamp"]).json()["same"] is True


def test_the_analysis_endpoints_return_empty_panels_on_an_empty_index(tmp_path, monkeypatch):
    """@brief 空库不是“混纪元”：三个分析端点都返回 200 的空面板，而不是 400。

    @param tmp_path pytest 提供的临时目录。
    @param monkeypatch pytest 的替换工具。
    """
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    monkeypatch.setenv("OBSERVE_CARRYOVER", str(tmp_path / "carry.json"))
    monkeypatch.setenv("TAIC_CATALOG_ROOT", str(tmp_path / "kernels"))
    with TestClient(app) as client:
        gap = client.get("/api/v2/gap")
        assert gap.status_code == 200
        body = gap.json()
        assert body["published_total"] == 0 and body["rows"] == [] and body["unpublished"] == 0
        assert body["protocol_epoch"] == "" and body["epochs"] == []
        committees = client.get("/api/v2/committees")
        assert committees.status_code == 200 and committees.json()["buckets"] == []
        noise = client.get("/api/v2/noise")
        assert noise.status_code == 200 and noise.json()["cv"]["n"] == 0


def test_editing_the_page_takes_effect_without_restarting(tmp_path, monkeypatch):
    """@brief 改 static/index.html 后不必重启进程：下一次请求就重读模板，两个槽位仍被填好。

    旧实现把 HTML 读成 import 期的模块常量，看板跑久了会一直吐进程启动那天的界面。

    @param tmp_path pytest 提供的临时目录，放临时模板与运行时路径。
    @param monkeypatch pytest 的替换工具，用来把模板路径指到临时文件。
    """
    from observe import app as observe_app

    template = tmp_path / "index.html"
    template.write_text(
        "<html><!--BUDGET_BANNER--><p>old page</p><!--OBSERVE_TOKEN--></html>", encoding="utf-8")
    monkeypatch.setattr(observe_app, "_PAGE_PATH", template)
    monkeypatch.setenv("AI_LOWERLING_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("OBSERVE_DB", str(tmp_path / "observe.sqlite"))
    monkeypatch.setenv("AI_LOWERLING_BOOKS", str(tmp_path / "books.sqlite"))
    with TestClient(app) as client:
        first = client.get("/").text
        assert "<p>old page</p>" in first
        assert "<!--BUDGET_BANNER-->" not in first and "<!--OBSERVE_TOKEN-->" not in first
        assert 'name="observe-token"' in first
        template.write_text(
            "<html><!--BUDGET_BANNER--><p>edited page, longer</p><!--OBSERVE_TOKEN--></html>", encoding="utf-8")
        second = client.get("/").text
        assert "edited page, longer" in second and "old page" not in second
