"""The dashboard sends an ETag and gzip, and answers If-None-Match with 304."""

from starlette.testclient import TestClient

from observe.app import app


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
