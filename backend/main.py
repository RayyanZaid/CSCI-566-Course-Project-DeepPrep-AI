import cgi
import json
import os
import tempfile
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from posture import (
    DEFAULT_MODEL_PATH,
    DEFAULT_POSE_LANDMARKER_PATH,
    DEFAULT_SCALER_PATH,
    analyze_posture_video,
)

HOST = "127.0.0.1"
PORT = 5000

BACKEND_DIR = Path(__file__).resolve().parent


def _build_frontend_result(analysis: dict) -> dict[str, str]:
    score = analysis.get("score", 0.0)
    metrics = analysis.get("metrics", {})
    body_language_feedback = analysis.get("feedback", "")

    engagement_feedback = (
        "This current demo estimates posture cues only, so eye contact and facial "
        "expression feedback are limited. Make sure your face is well lit and "
        "positioned near the camera for stronger engagement signals in future iterations."
    )

    content_feedback = (
        "This prototype does not yet score interview answer content. It currently "
        "focuses on non-verbal posture cues from the uploaded video."
    )

    return {
        "Final Score": f"{score:.2f}",
        "Body Language and Posture Feedback": (
            f"Posture score: {score:.2f}. {body_language_feedback} "
            f"Average head tilt: {metrics.get('average_head_tilt', 0.0):.2f}. "
            f"Average shoulder angle: {metrics.get('average_shoulder_angle', 0.0):.2f}. "
            f"Average forward lean: {metrics.get('average_forward_lean', 0.0):.3f}."
        ),
        "Engagement (Eye Contact and Facial Expressions) Feedback": engagement_feedback,
        "Interview Response Content Feedback": content_feedback,
    }


class AnalyzeHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json({}, status=HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        self._send_json(
            {"error": "Route not found. Use POST /analyze or GET /health."},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self._send_json({"error": "Route not found."}, status=HTTPStatus.NOT_FOUND)
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(
                {"error": "Expected multipart/form-data with a `video` file."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        video_field = form["video"] if "video" in form else None
        if video_field is None or not getattr(video_field, "file", None):
            self._send_json(
                {"error": "No video file was uploaded under the `video` field."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        original_name = os.path.basename(video_field.filename or "upload.mp4")
        suffix = Path(original_name).suffix or ".mp4"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_field.file.read())

            analysis = analyze_posture_video(
                video_path=temp_path,
                pose_landmarker_path=str(BACKEND_DIR / "pose_landmarker.task"),
                model_path=str(BACKEND_DIR / "posture_model.h5"),
                scaler_path=str(BACKEND_DIR / "scaler.save"),
            )
            result = _build_frontend_result(analysis)

            self._send_json(
                {
                    "success": True,
                    "analysis": analysis,
                    "result": result,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive API boundary
            print("Error while processing /analyze request:")
            traceback.print_exc()
            self._send_json(
                {"success": False, "error": str(exc)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def run_server(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), AnalyzeHandler)
    print(f"Posture backend listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
