/**
 * Render segmentation masks, bbox outlines and labels.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array<Object>} boxes  — each box has { label, probability, color, bounding, mask, protoW, protoH }
 */
export const renderBoxes = (ctx, boxes) => {
  const font = `${Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )}px Arial`;
  ctx.font = font;
  ctx.textBaseline = "top";

  // ── Pass 1: draw instance masks ──────────────────────────────────────────
  boxes.forEach((box) => {
    if (!box.mask) return;
    const { mask, protoW, protoH, color } = box;
    const rgba = Colors.hexToRgba(color, 140); // semi-transparent fill
    if (!rgba) return;

    // Paint mask into a tiny (protoW × protoH) offscreen canvas
    const offscreen = document.createElement("canvas");
    offscreen.width  = protoW;
    offscreen.height = protoH;
    const offCtx   = offscreen.getContext("2d");
    const imgData  = offCtx.createImageData(protoW, protoH);

    for (let p = 0; p < protoW * protoH; p++) {
      if (mask[p] > 0.5) {
        const idx = p * 4;
        imgData.data[idx]     = rgba[0];
        imgData.data[idx + 1] = rgba[1];
        imgData.data[idx + 2] = rgba[2];
        imgData.data[idx + 3] = rgba[3];
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    // Scale 160×160 → full canvas (640×640) — aligns perfectly with model coords
    ctx.drawImage(offscreen, 0, 0, ctx.canvas.width, ctx.canvas.height);
  });

  // ── Pass 2: draw bbox outlines + labels ──────────────────────────────────
  boxes.forEach((box) => {
    const klass = box.label;
    const color = box.color;
    const score = (box.probability * 100).toFixed(1);
    const [x1, y1, width, height] = box.bounding;

    // bbox outline
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5);
    ctx.strokeRect(x1, y1, width, height);

    // label background
    ctx.fillStyle = color;
    const textWidth  = ctx.measureText(`${klass} - ${score}%`).width;
    const textHeight = parseInt(font, 10);
    const yText = y1 - (textHeight + ctx.lineWidth);
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    );

    // label text
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`${klass} - ${score}%`, x1 - 1, yText < 0 ? 1 : yText + 1);
  });
};

export class Colors {
  constructor() {
    // Custom colors for watermelon (green), melon (yellow), pumpkin (orange)
    this.palette = [
      "#2ECC40", // watermelon - green
      "#FFDC00", // melon - yellow
      "#FF851B", // pumpkin - orange
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16), alpha]
      : null;
  };
}
