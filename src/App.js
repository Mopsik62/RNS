import React, { useState, useRef, useEffect } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import "./style/App.css";

const MODEL_OPTIONS = [
  { label: "Run 1 — baseline (yolov8n, lr=0.01)",         file: "run1_baseline.onnx" },
  { label: "Run 2 — low lr (yolov8n, lr=0.001)",          file: "run2_low_lr.onnx" },
  { label: "Run 3 — augmented (yolov8n, mixup, degrees)", file: "run3_augmented.onnx" },
  { label: "Run 4 — large model (yolov8s, lr=0.01)",      file: "run4_yolov8s.onnx" },
];

const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.25;

const App = () => {
  const [selectedIdx, setSelectedIdx]     = useState(0);
  const [session, setSession]             = useState(null);
  const [loading, setLoading]             = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage]                 = useState(null);
  const [detectionInfo, setDetectionInfo] = useState(null);
  const inputImage = useRef(null);
  const imageRef   = useRef(null);
  const canvasRef  = useRef(null);
  const cvReady    = useRef(false);

  const loadModel = async (modelFile) => {
    setLoading({ text: `Loading ${modelFile}…`, progress: null });
    setSession(null);
    const net = await InferenceSession.create(`./${modelFile}`);
    // Warm up
    const dummy = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await net.run({ images: dummy });
    setSession({ net });
    setLoading(null);
  };

  // OpenCV init → load first model
  cv["onRuntimeInitialized"] = async () => {
    cvReady.current = true;
    await loadModel(MODEL_OPTIONS[0].file);
  };

  // Reload model on dropdown change
  useEffect(() => {
    if (!cvReady.current) return;
    // Reset canvas and image
    if (canvasRef.current) {
      canvasRef.current.getContext("2d")
        .clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    if (image) {
      URL.revokeObjectURL(image);
      setImage(null);
      setDetectionInfo(null);
      if (imageRef.current)  imageRef.current.src = "#";
      if (inputImage.current) inputImage.current.value = "";
    }
    loadModel(MODEL_OPTIONS[selectedIdx].file);
  }, [selectedIdx]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}

      <div className="header">
        <h1>YOLOv8 Object Detection App</h1>
        <p>
          Классы:{" "}
          <strong style={{ color: "#2ECC40" }}>spacecraft</strong>,{" "}
          <strong style={{ color: "#e6b800" }}>rocket</strong>,{" "}
          <strong style={{ color: "#FF851B" }}>satellite</strong>
        </p>

        <div style={{ margin: "10px 0" }}>
          <label htmlFor="model-select" style={{ marginRight: 8, fontWeight: "bold" }}>
            Модель:
          </label>
          <select
            id="model-select"
            value={selectedIdx}
            disabled={!!loading}
            onChange={(e) => setSelectedIdx(Number(e.target.value))}
            style={{ padding: "4px 8px", fontSize: 14, borderRadius: 4 }}
          >
            {MODEL_OPTIONS.map((opt, i) => (
              <option key={opt.file} value={i}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <p>
          Файл: <code className="code">{MODEL_OPTIONS[selectedIdx].file}</code>
        </p>
        {detectionInfo && (
          <p style={{ fontSize: "12px", color: "#555" }}>{detectionInfo}</p>
        )}
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            if (!session) {
              alert("Модель ещё загружается, подождите");
              return;
            }
            setDetectionInfo("Running detection…");
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              scoreThreshold,
              modelInputShape
            )
              .then((count) => setDetectionInfo(`Detected: ${count ?? "?"} object(s)`))
              .catch((err) => {
                console.error(err);
                setDetectionInfo("Error: " + err.message);
              });
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          if (image) URL.revokeObjectURL(image);
          setDetectionInfo(null);
          const url = URL.createObjectURL(e.target.files[0]);
          imageRef.current.src = url;
          setImage(url);
        }}
      />

      <div className="btn-container">
        <button onClick={() => inputImage.current.click()}>Open local image</button>
        {image && (
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
              setDetectionInfo(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
