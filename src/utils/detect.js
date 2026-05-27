import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes, Colors } from "./renderBox";
import labels from "./labels.json";

const colors = new Colors();
const numClass = labels.length;

export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape);
  const outputs = await session.net.run({ images: tensor });

  // Detection ONNX: output0 shape [1, 4+nc, 8400]
  // Маски берём синтетические (эллипс по bbox) — костыль без seg-модели
  const output0 = outputs.output0;
  const [, , anchors] = output0.dims;

  // Proto-размер для синтетической маски
  const protoW = 160, protoH = 160;

  // ── Parse candidates ──────────────────────────────────────────────────────
  const candidates = [];
  for (let i = 0; i < anchors; i++) {
    let maxScore = 0;
    let label = 0;
    for (let c = 0; c < numClass; c++) {
      const s = output0.data[(4 + c) * anchors + i];
      if (s > maxScore) { maxScore = s; label = c; }
    }
    if (maxScore < scoreThreshold) continue;

    const cx = output0.data[0 * anchors + i];
    const cy = output0.data[1 * anchors + i];
    const w  = output0.data[2 * anchors + i];
    const h  = output0.data[3 * anchors + i];
    candidates.push({ cx, cy, w, h, score: maxScore, label });
  }

  // ── NMS ──────────────────────────────────────────────────────────────────
  candidates.sort((a, b) => b.score - a.score);
  const kept = jsNMS(candidates.slice(0, topk * 10), iouThreshold).slice(0, topk);

  // ── GrabCut: готовим изображение в display-пространстве (простой ресайз) ────
  // Простой ресайз до 160×160 без letterbox — координаты совпадут с display bbox
  const gcMat = prepareGrabCutMat(image, protoW, protoH);

  // ── Build boxes + GrabCut-маска ───────────────────────────────────────────
  const boxes = kept.map((d) => {
    // Display-координаты bbox — вычисляем первыми
    const box = overflowBoxes([d.cx - d.w / 2, d.cy - d.h / 2, d.w, d.h], maxSize);
    const [x, y, bw, bh] = overflowBoxes(
      [
        Math.floor(box[0] * xRatio),
        Math.floor(box[1] * yRatio),
        Math.floor(box[2] * xRatio),
        Math.floor(box[3] * yRatio),
      ],
      maxSize
    );

    // GrabCut получает display-координаты — маска окажется в том же пространстве
    const mask =
      runGrabCut(gcMat, x, y, bw, bh, protoW, protoH, modelWidth, modelHeight) ??
      makeEllipseMask(x, y, bw, bh, protoW, protoH, modelWidth, modelHeight);

    return {
      label:       labels[d.label],
      probability: d.score,
      color:       colors.get(d.label),
      bounding:    [x, y, bw, bh],
      mask,
      protoW,
      protoH,
    };
  });

  gcMat.delete();
  renderBoxes(ctx, boxes);
  input.delete();
  return boxes.length;
};

// ── GrabCut: подготовка изображения ─────────────────────────────────────────
// Простой ресайз до protoW×protoH (160×160) БЕЗ letterbox.
// Координаты в этом пространстве = display-координаты × 0.25,
// что совпадает с bbox после xRatio/yRatio трансформации.
function prepareGrabCutMat(source, protoW, protoH) {
  const mat   = cv.imread(source);
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);
  mat.delete();

  const matSmall = new cv.Mat();
  cv.resize(matC3, matSmall, new cv.Size(protoW, protoH));
  matC3.delete();

  return matSmall; // вызывающий обязан вызвать .delete()
}

// ── GrabCut: сегментация одного объекта ─────────────────────────────────────
// x1, y1, bw, bh — display-координаты bbox (в пространстве канваса 640×640)
// Возвращает Float32Array [protoW×protoH] или null при ошибке
function runGrabCut(gcMat, x1, y1, bw, bh, protoW, protoH, canvasW, canvasH) {
  // Display-координаты → proto-координаты (160×160)
  const scale = protoW / canvasW; // 0.25
  const gx1 = Math.max(0,      Math.round(x1 * scale));
  const gy1 = Math.max(0,      Math.round(y1 * scale));
  const gx2 = Math.min(protoW, Math.round((x1 + bw) * scale));
  const gy2 = Math.min(protoH, Math.round((y1 + bh) * scale));
  const rw  = gx2 - gx1;
  const rh  = gy2 - gy1;
  if (rw < 4 || rh < 4) return null;

  const maskMat = new cv.Mat();
  const bgModel = new cv.Mat();
  const fgModel = new cv.Mat();
  try {
    cv.grabCut(
      gcMat, maskMat,
      new cv.Rect(gx1, gy1, rw, rh),
      bgModel, fgModel,
      3,                       // итерации
      cv.GC_INIT_WITH_RECT
    );

    const result = new Float32Array(protoW * protoH);
    const data   = maskMat.data; // GC_BGD=0, GC_FGD=1, GC_PR_BGD=2, GC_PR_FGD=3
    for (let i = 0; i < data.length; i++) {
      if (data[i] === 1 || data[i] === 3) result[i] = 1.0;
    }
    return result;
  } catch (_) {
    return null; // fallback к эллипсу
  } finally {
    maskMat.delete();
    bgModel.delete();
    fgModel.delete();
  }
}

// ── Эллипс-маска (fallback если GrabCut не сработал) ────────────────────────
function makeEllipseMask(x1, y1, bw, bh, protoW, protoH, canvasW, canvasH) {
  const sx  = protoW / canvasW;
  const sy  = protoH / canvasH;
  const ecx = (x1 + bw / 2) * sx;
  const ecy = (y1 + bh / 2) * sy;
  const rx  = Math.max((bw / 2) * sx, 0.5);
  const ry  = Math.max((bh / 2) * sy, 0.5);
  const mask = new Float32Array(protoW * protoH);
  for (let y = 0; y < protoH; y++) {
    for (let x = 0; x < protoW; x++) {
      const dx = (x + 0.5 - ecx) / rx;
      const dy = (y + 0.5 - ecy) / ry;
      if (dx * dx + dy * dy <= 1.0) mask[y * protoW + x] = 1.0;
    }
  }
  return mask;
}

// ── JS NMS ──────────────────────────────────────────────────────────────────
function jsNMS(boxes, iouThreshold) {
  const suppressed = new Array(boxes.length).fill(false);
  const kept = [];
  for (let i = 0; i < boxes.length; i++) {
    if (suppressed[i]) continue;
    kept.push(boxes[i]);
    for (let j = i + 1; j < boxes.length; j++) {
      if (!suppressed[j] && iouCxcywh(boxes[i], boxes[j]) > iouThreshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

function iouCxcywh(a, b) {
  const ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  const ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  const bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  const bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// ── Preprocessing (без изменений) ───────────────────────────────────────────
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source);
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3);
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  const maxSize = Math.max(matC3.rows, matC3.cols);
  const xPad = maxSize - matC3.cols, xRatio = maxSize / matC3.cols;
  const yPad = maxSize - matC3.rows, yRatio = maxSize / matC3.rows;
  const matPad = new cv.Mat();
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT);

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0,
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0),
    true,
    false
  );

  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    width =
      width % stride >= stride / 2
        ? (Math.floor(width / stride) + 1) * stride
        : Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    height =
      height % stride >= stride / 2
        ? (Math.floor(height / stride) + 1) * stride
        : Math.floor(height / stride) * stride;
  }
  return [width, height];
};

const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};
