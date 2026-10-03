async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_returns_ready_when_db_reachable(client):
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


async def test_request_id_is_echoed(client):
    response = await client.get("/health", headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"


async def test_unknown_route_uses_standard_error_body(client):
    response = await client.get("/does-not-exist", headers={"X-Request-ID": "req-404"})
    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "HTTP_ERROR", "message": "Not Found", "request_id": "req-404"}
    }
