from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def _login(email: str = "instructor@classroom.local", password: str = "classroom123") -> str:
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200
    data = response.json()
    assert data["token"]
    assert data["user"]["email"] == email
    return data["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_local_auth_seeded_accounts_and_bad_password() -> None:
    token = _login()
    response = client.get("/api/auth/me", headers=_auth(token))
    assert response.status_code == 200
    assert response.json()["user"]["role"] == "instructor"

    bad = client.post("/api/auth/login", json={"email": "instructor@classroom.local", "password": "wrong"})
    assert bad.status_code == 401


def test_subject_create_list_delete_requires_local_auth() -> None:
    unauthorized = client.get("/api/subjects")
    assert unauthorized.status_code == 401

    token = _login()
    payload = {
        "name": "API Smoke Subject",
        "code": "SMK101",
        "description": "Created by local API smoke test",
        "totalStudents": 12,
    }
    created = client.post("/api/subjects", json=payload, headers=_auth(token))
    assert created.status_code == 200
    subject = created.json()
    assert subject["name"] == payload["name"]
    assert subject["code"] == payload["code"]

    listed = client.get("/api/subjects", headers=_auth(token))
    assert listed.status_code == 200
    assert any(item["id"] == subject["id"] for item in listed.json())

    deleted = client.delete(f"/api/subjects/{subject['id']}", headers=_auth(token))
    assert deleted.status_code == 200


def test_run_history_requires_auth_and_file_download_stays_inside_outputs() -> None:
    unauthenticated = client.get("/api/dates")
    assert unauthenticated.status_code == 401

    token = _login()
    dates = client.get("/api/dates", headers=_auth(token))
    assert dates.status_code == 200
    assert isinstance(dates.json(), list)

    outside = client.get("/api/files", params={"path": str(Path("app/api/main.py"))})
    assert outside.status_code == 403
