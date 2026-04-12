import React from "react";

interface ResultProps {
  result: Record<string, string>; // 👈 change from strict keys to general
}

const Result: React.FC<ResultProps> = ({ result }) => {
  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h1 style={{ marginBottom: "24px" }}>Your Interview Result</h1>

      {result["Body Language and Posture Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Body Language and Posture</h2>
          <p>{result["Body Language and Posture Feedback"]}</p>
        </section>
      )}

      {result["Engagement (Eye Contact and Facial Expressions) Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Engagement & Eye Contact</h2>
          <p>
            {result["Engagement (Eye Contact and Facial Expressions) Feedback"]}
          </p>
        </section>
      )}

      {result["Interview Response Content Feedback"] && (
        <section style={{ marginBottom: "16px" }}>
          <h2>Interview Response Content</h2>
          <p>{result["Interview Response Content Feedback"]}</p>
        </section>
      )}
    </div>
  );
};

export default Result;
