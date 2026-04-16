import { CSSProperties, useRef, useState } from "react";
import { FaTrash as TrashIcon, FaUpload as UploadIcon } from "react-icons/fa";

import InterviewImage from "../assets/InterviewImage.png";
import Result from "./Result";
import Spinner from "./Spinner";

const API_BASE_URL =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:5000";

const styles: Record<string, CSSProperties> = {
  submitButton: {
    padding: "16px 32px",
    fontSize: "18px",
    fontWeight: "bold",
    borderRadius: "8px",
    border: "none",
    backgroundColor: "#007bff",
    color: "#fff",
    cursor: "pointer",
  },
};

function UploadVideo() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<
    "pending" | "uploading" | "success" | "error"
  >("pending");

  const [videoURL, setVideoURL] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [result, setResult] = useState<any>(null); // 👈 treat it as any JSON object

  const handleUpload = async () => {
    if (!file) return;

    setStatus("uploading");

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      console.log("Response status: ", response.status);

      if (!response.ok) {
        const errorBody = await response.text();
        console.log("Response.ok: ", response.ok);
        console.log("Error: ", response.statusText);
        console.log("Error body: ", errorBody);
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      const parsedResult =
        typeof data.result === "string"
          ? JSON.parse(
              data.result.replace(/```json\n/, "").replace(/```/, "").trim()
            )
          : data.result;

      setResult(parsedResult);

      console.log("Result: ", parsedResult);
      setStatus("success");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleUpload();
      }}
      onReset={() => {
        setStatus("pending");
        setFile(null);
        setVideoURL(null);
        setResult(null);
      }}
    >
      {status === "uploading" ? (
        <div
          style={{ display: "flex", justifyContent: "center", padding: "2rem" }}
        >
          <Spinner />
        </div>
      ) : (
        <>
          {!file && (
            <div>
              <h1 style={styles.heading}>Upload Your Interview Video Here</h1>
              <img
                src={InterviewImage}
                alt="Interview"
                style={{ maxWidth: "100%", height: "auto" }}
              />
            </div>
          )}

          <div>
            <div style={styles.content}>
              {status === "success" && result && <Result result={result} />}
              {status === "error" && (
                <p style={styles.errorText}>File upload failed!</p>
              )}

              <div style={{ height: "50px" }} />

              {videoURL && (
                <video
                  src={videoURL}
                  controls
                  style={{
                    width: "100%",
                    maxHeight: "400px",
                    marginTop: "1rem",
                  }}
                />
              )}
              <div style={{ height: "50px" }} />
              <div
                style={styles.uploadBox}
                onClick={() => inputRef.current?.click()}
              >
                <UploadIcon size={32} />
              </div>

              <h2 style={styles.uploadText}>
                Drop your files here,
                <br /> or browse
              </h2>
            </div>
          </div>

          <input
            type="file"
            onChange={(e) => {
              setStatus("pending");
              const selectedFile = e.target.files?.[0] || null;
              setFile(selectedFile);
              if (selectedFile) {
                setVideoURL(URL.createObjectURL(selectedFile));
              } else {
                setVideoURL(null);
              }
            }}
            hidden
            ref={inputRef}
          />

          {file && (
            <>
              <div style={styles.fileList}>
                <div style={styles.fileElement}>
                  <div style={styles.filler} />
                  {status === "pending" && (
                    <button
                      style={styles.trash}
                      onClick={(e) => {
                        e.preventDefault();
                        if (inputRef.current) {
                          inputRef.current.value = "";
                        }
                        setFile(null);
                        setVideoURL(null);
                      }}
                    >
                      <TrashIcon size={32} />
                    </button>
                  )}
                </div>
              </div>

              <div style={styles.uploadButtonContainer}>
                {status === "success" ? (
                  <button style={styles.submitButton} type="reset">
                    Reset
                  </button>
                ) : (
                  <div>
                    <div style={{ height: "50px" }} />
                    <button style={styles.submitButton} type="submit">
                      Submit
                    </button>
                  </div>
                )}
              </div>
            </>
          )}
        </>
      )}
    </form>
  );
}

export default UploadVideo;
