// Freehand brush over an image.
//
// The overlay canvases are given exactly the CSS box of the <img>, and every
// pointer position is converted with getBoundingClientRect() into 0..1 image
// coordinates. Browser zoom and Windows display scaling change the rect and the
// devicePixelRatio together, so the normalised position is always exact.
//
// Python owns the mask: each finished stroke is sent as {id, erase, r, points}
// and stays "pending" here (drawn on the overlay) until Python reports it in
// ack_id and sends back an image that already contains it.

const img = document.getElementById("image");
const paint = document.getElementById("paint");
const cursor = document.getElementById("cursor");
const paintCtx = paint.getContext("2d");
const cursorCtx = cursor.getContext("2d");

const BRUSH_COLOR = "rgba(255, 0, 0, 0.45)";
const ERASER_COLOR = "rgba(255, 255, 255, 0.55)";

let args = null;
let pending = [];
let current = null;
let lastId = 0;
let loadedSrc = null;

function nextId() {
  lastId = Math.max(Date.now() * 1000, lastId + 1);
  return lastId;
}

function syncSize() {
  const rect = img.getBoundingClientRect();
  if (!rect.width || !rect.height) {
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  for (const c of [paint, cursor]) {
    c.style.width = rect.width + "px";
    c.style.height = rect.height + "px";
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
  }
  redraw();
  Streamlit.setFrameHeight(Math.ceil(rect.height));
}

function toNorm(event) {
  const rect = paint.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width;
  const y = (event.clientY - rect.top) / rect.height;
  return [Math.min(Math.max(x, 0), 1), Math.min(Math.max(y, 0), 1)];
}

function drawStroke(ctx, stroke) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const radius = stroke.r * W;
  ctx.fillStyle = ctx.strokeStyle = stroke.erase ? ERASER_COLOR : BRUSH_COLOR;
  const pts = stroke.points;
  if (pts.length === 1) {
    ctx.beginPath();
    ctx.arc(pts[0][0] * W, pts[0][1] * H, radius, 0, 2 * Math.PI);
    ctx.fill();
    return;
  }
  ctx.lineWidth = 2 * radius;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(pts[0][0] * W, pts[0][1] * H);
  for (let i = 1; i < pts.length; i++) {
    ctx.lineTo(pts[i][0] * W, pts[i][1] * H);
  }
  ctx.stroke();
}

function redraw() {
  paintCtx.clearRect(0, 0, paint.width, paint.height);
  for (const s of pending) {
    drawStroke(paintCtx, s);
  }
  if (current) {
    drawStroke(paintCtx, current);
  }
}

function drawCursor(event) {
  cursorCtx.clearRect(0, 0, cursor.width, cursor.height);
  if (!event || !args) {
    return;
  }
  const [x, y] = toNorm(event);
  cursorCtx.beginPath();
  cursorCtx.arc(x * cursor.width, y * cursor.height, args.radius * cursor.width, 0, 2 * Math.PI);
  cursorCtx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  cursorCtx.strokeStyle = args.erase ? "rgba(255, 255, 255, 0.9)" : "rgba(255, 0, 0, 0.9)";
  cursorCtx.stroke();
}

function sendPending() {
  Streamlit.setComponentValue({
    strokes: pending.map((s) => ({
      id: s.id,
      erase: s.erase,
      r: s.r,
      points: s.points.map(([x, y]) => [Math.round(x * 1e5) / 1e5, Math.round(y * 1e5) / 1e5]),
    })),
  });
}

paint.addEventListener("pointerdown", (event) => {
  if (!args || event.button !== 0) {
    return;
  }
  event.preventDefault();
  paint.setPointerCapture(event.pointerId);
  current = {id: nextId(), erase: !!args.erase, r: args.radius, points: [toNorm(event)]};
  redraw();
});

paint.addEventListener("pointermove", (event) => {
  drawCursor(event);
  if (!current) {
    return;
  }
  const p = toNorm(event);
  const last = current.points[current.points.length - 1];
  const rect = paint.getBoundingClientRect();
  if (Math.hypot((p[0] - last[0]) * rect.width, (p[1] - last[1]) * rect.height) >= 1) {
    current.points.push(p);
    redraw();
  }
});

function finishStroke(event) {
  if (!current) {
    return;
  }
  if (event.type === "pointerup") {
    const p = toNorm(event);
    const last = current.points[current.points.length - 1];
    if (p[0] !== last[0] || p[1] !== last[1]) {
      current.points.push(p);
    }
  }
  pending.push(current);
  current = null;
  redraw();
  sendPending();
}

paint.addEventListener("pointerup", finishStroke);
paint.addEventListener("pointercancel", finishStroke);
paint.addEventListener("pointerleave", () => drawCursor(null));

function dropAcknowledged() {
  const ack = args ? args.ack_id || 0 : 0;
  pending = pending.filter((s) => s.id > ack);
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, (event) => {
  args = event.detail.args;
  img.style.width = args.width ? args.width + "px" : "";

  if (args.src !== loadedSrc) {
    loadedSrc = args.src;
    // Keep pending strokes visible until the image that contains them has loaded.
    img.onload = () => {
      dropAcknowledged();
      syncSize();
    };
    img.src = args.src;
  } else {
    dropAcknowledged();
    syncSize();
  }
});

new ResizeObserver(syncSize).observe(img);
window.addEventListener("resize", syncSize);

Streamlit.setComponentReady();
