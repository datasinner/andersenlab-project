"""Uploading a tariff PDF over HTTP (ingested in the background) and the
optional API key."""

from app.config import settings
from app.llm.client import FakeLLMClient
from tests.integration.conftest import TNPA_CATALOGUE, TNPA_PDF, TNPA_PROFILE


def _upload(client, content: bytes, name: str = "tariff.pdf", **params):
    return client.post(
        "/v1/documents", files={"file": (name, content, "application/pdf")}, params=params
    )


async def test_upload_ingests_in_the_background(client, app_instance):
    llm: FakeLLMClient = app_instance.state.llm_client
    llm.script("document_profile", TNPA_PROFILE)
    llm.script("charge_catalogue", TNPA_CATALOGUE)

    response = await _upload(client, TNPA_PDF.read_bytes(), "../../etc/tnpa.pdf")

    assert response.status_code == 202, response.text
    body = response.json()
    assert (body["status"], body["created"]) == ("parsing", True)
    # The background task has run by the time the test client returns.
    detail = (await client.get(f"/v1/documents/{body['document_id']}")).json()
    assert detail["status"] == "ready"
    assert detail["source_filename"] == "tnpa.pdf"  # no path from the client
    assert detail["charges"] == 3
    assert detail["sections"] > 90

    again = await _upload(client, TNPA_PDF.read_bytes())
    assert again.status_code == 200
    assert again.json() == {"document_id": body["document_id"], "status": "ready", "created": False}


async def test_upload_rejects_non_pdfs_and_oversized_files(client, monkeypatch):
    not_pdf = await _upload(client, b"just text", "notes.txt")
    assert not_pdf.status_code == 415
    assert not_pdf.json()["error"]["code"] == "NOT_A_PDF"

    monkeypatch.setattr(settings, "max_upload_mb", 1)
    too_big = await _upload(client, b"%PDF" + b"0" * (1024 * 1024))
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


async def test_api_key_protects_v1_routes_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_auth_key", "s3cret")

    missing = await client.get("/v1/documents")
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "UNAUTHORIZED"
    wrong = await client.get("/v1/documents", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    right = await client.get("/v1/documents", headers={"X-API-Key": "s3cret"})
    assert right.status_code == 200
    assert (await client.get("/health")).status_code == 200


async def test_openapi_declares_the_api_key_scheme(client):
    schema = (await client.get("/openapi.json")).json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
    upload = schema["paths"]["/v1/documents"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]
