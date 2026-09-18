import unittest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAdminAPI(unittest.TestCase):

    def test_model_status(self):
        response = client.get("/api/model-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("status", data["data"])
        self.assertIn("model", data["data"])

    def test_export_dataset_unauthorized(self):
        response = client.get("/api/admin/export-dataset")
        self.assertEqual(response.status_code, 401)

    def test_export_dataset_forbidden(self):
        response = client.get(
            "/api/admin/export-dataset", 
            headers={"Authorization": "Bearer wrongtoken"}
        )
        self.assertEqual(response.status_code, 403)

    def test_export_dataset_success(self):
        response = client.get(
            "/api/admin/export-dataset", 
            headers={"Authorization": "Bearer supersecretadmin"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/jsonl"))


if __name__ == "__main__":
    unittest.main()
