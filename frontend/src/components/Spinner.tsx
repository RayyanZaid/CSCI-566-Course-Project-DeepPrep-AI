import { ClipLoader } from "react-spinners";

const Spinner = () => {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
      }}
    >
      <h1 style={{ marginRight: "1rem" }}>Analyzing your interview...</h1>
      <ClipLoader color="#3498db" size={80} />
    </div>
  );
};

export default Spinner;
