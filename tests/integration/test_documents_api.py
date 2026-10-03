import uuid


async def test_list_and_detail(client, ingested_tnpa):
    listing = (await client.get("/v1/documents")).json()
    assert len(listing) == 1
    summary = listing[0]
    assert summary["id"] == str(ingested_tnpa.document_id)
    assert summary["status"] == "ready"
    assert summary["ports"] == ["Durban"]

    detail = (await client.get(f"/v1/documents/{ingested_tnpa.document_id}")).json()
    assert detail["currency"] == "ZAR"
    assert detail["vat_rate"] == "0.1500"
    assert detail["effective_from"] == "2024-04-01"
    assert detail["ports"] == [{"name": "Durban", "aliases": ["Port of Durban"]}]
    assert detail["sections"] == ingested_tnpa.sections
    assert detail["chunks"] == ingested_tnpa.chunks
    assert detail["charges"] == 3
    assert detail["error"] is None


async def test_charges(client, ingested_tnpa):
    response = await client.get(f"/v1/documents/{ingested_tnpa.document_id}/charges")
    assert response.status_code == 200
    assert [charge["charge_id"] for charge in response.json()] == [
        "light_dues",
        "towage",
        "dry_bulk_cargo_dues",
    ]
    assert response.json()[1]["section_refs"] == ["3.6"]


async def test_section(client, ingested_tnpa):
    response = await client.get(f"/v1/documents/{ingested_tnpa.document_id}/sections/4.1")
    body = response.json()
    assert response.status_code == 200
    assert body["title"] == "4.1 PORT FEES ON VESSELS"
    assert body["children"] == [
        {"ref": "4.1.1", "title": "4.1.1 PORT DUES"},
        {"ref": "4.1.2", "title": "4.1.2 BERTH DUES"},
    ]
    assert "gross tonnage" in body["text"]


async def test_unknown_document_and_section_are_404(client, ingested_tnpa):
    unknown = await client.get(f"/v1/documents/{uuid.uuid4()}")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    charges = await client.get(f"/v1/documents/{uuid.uuid4()}/charges")
    assert charges.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    section = await client.get(f"/v1/documents/{ingested_tnpa.document_id}/sections/99.9")
    assert section.status_code == 404
    assert section.json()["error"]["code"] == "SECTION_NOT_FOUND"


async def test_malformed_document_id_is_a_validation_error(client):
    response = await client.get("/v1/documents/not-a-uuid")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
