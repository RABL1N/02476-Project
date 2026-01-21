from locust import HttpUser, task, between
from pathlib import Path


class FastAPIUser(HttpUser):
    # Wait time between requests (DTU-style)
    wait_time = between(1, 2)

    @task
    def predict(self):
        image_path = Path("load_tests/test_image.png")

        with image_path.open("rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("test.png", f, "image/png")},
            )
