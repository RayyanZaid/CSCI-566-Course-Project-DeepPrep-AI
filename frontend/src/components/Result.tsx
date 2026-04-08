import React from "react";

interface ResultProps {
  result: Record<string, string>; // 👈 change from strict keys to general
}

const Result: React.FC<ResultProps> = ({ result }) => {
  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h1 style={{ marginBottom: "24px" }}>Your Interview Result</h1>

      {result["Final Score"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Final Score</h2>
          <p>{result["Final Score"]}</p>
        </section>
      )}

      {result["Facial Gesture Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Facial Gesture Feedback</h2>
          <p>{result["Facial Gesture Feedback"]}</p>
        </section>
      )}

      {result["Prosody Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Prosody Feedback</h2>
          <p>{result["Prosody Feedback"]}</p>
        </section>
      )}

      {result["Eye Contact Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Eye Contact Feedback</h2>
          <p>{result["Eye Contact Feedback"]}</p>
        </section>
      )}

      {result["Posture Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Posture Feedback</h2>
          <p>{result["Posture Feedback"]}</p>
        </section>
      )}
    </div>
  );
};

export default Result;
