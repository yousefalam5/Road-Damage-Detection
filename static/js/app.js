/* =====================================================
   RoadScan AI – JavaScript
   Handles: drag-drop upload, API call, results render
===================================================== */

"use strict";

// ─── Class colour map (must match Flask/CSS) ─────────────────
const CLASS_COLORS = {
  alligator:    "#f97316",
  block:        "#a855f7",
  crack:        "#22c55e",
  edge:         "#ec4899",
  longitudinal: "#06b6d4",
  pothole:      "#ef4444",
  transverse:   "#eab308",
};

// ─── DOM refs ────────────────────────────────────────────────
const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const browseBtn      = document.getElementById("browseBtn");
const previewWrap    = document.getElementById("previewWrap");
const previewImg     = document.getElementById("previewImg");
const detectBtn      = document.getElementById("detectBtn");
const clearBtn       = document.getElementById("clearBtn");
const resultsPlaceholder = document.getElementById("resultsPlaceholder");
const loadingState   = document.getElementById("loadingState");
const resultsContent = document.getElementById("resultsContent");
const resultImg      = document.getElementById("resultImg");
const detCount       = document.getElementById("detCount");
const detSpeed       = document.getElementById("detSpeed");
const detectionList  = document.getElementById("detectionList");
const downloadBtn    = document.getElementById("downloadBtn");

const confSlider     = document.getElementById("confSlider");
const confVal        = document.getElementById("confVal");
const iouSlider      = document.getElementById("iouSlider");
const iouVal         = document.getElementById("iouVal");
const sampleCards    = document.querySelectorAll(".sample-card");

let selectedFile     = null;
let lastAnnotatedB64 = null;
let sliderDebounceTimer = null;

// ─── Slider Event Handlers ───────────────────────────────────

if (confSlider && confVal) {
  confSlider.addEventListener("input", (e) => {
    confVal.textContent = `${e.target.value}%`;
    triggerDebouncedDetection();
  });
}

if (iouSlider && iouVal) {
  iouSlider.addEventListener("input", (e) => {
    iouVal.textContent = `${e.target.value}%`;
    triggerDebouncedDetection();
  });
}

function triggerDebouncedDetection() {
  if (!selectedFile) return;
  clearTimeout(sliderDebounceTimer);
  sliderDebounceTimer = setTimeout(() => {
    runDetection();
  }, 400);
}

// ─── Sample Images Click Handlers ────────────────────────────

sampleCards.forEach((card) => {
  card.addEventListener("click", async () => {
    const filename = card.getAttribute("data-filename");
    if (!filename) return;

    sampleCards.forEach(c => c.classList.remove("active-sample"));
    card.classList.add("active-sample");

    showLoading();
    // Scroll to detector smoothly
    document.getElementById("detector")?.scrollIntoView({ behavior: "smooth" });

    try {
      const imgUrl = `/static/img/samples/${filename}`;
      const resp = await fetch(imgUrl);
      if (!resp.ok) throw new Error("Could not load sample picture");
      const blob = await resp.blob();
      const file = new File([blob], filename, { type: blob.type || "image/jpeg" });

      selectedFile = file;
      previewImg.src = URL.createObjectURL(blob);
      dropzone.classList.add("hidden");
      previewWrap.classList.remove("hidden");

      await runDetection();
    } catch (err) {
      console.error(err);
      showPlaceholder();
      alert(`Could not load sample image: ${err.message}`);
    }
  });
});

// ─── Upload handlers ─────────────────────────────────────────

browseBtn.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click",  () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload an image file (PNG, JPG, JPEG).");
    return;
  }
  sampleCards.forEach(c => c.classList.remove("active-sample"));
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    dropzone.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    showPlaceholder();
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", () => {
  selectedFile     = null;
  lastAnnotatedB64 = null;
  fileInput.value  = "";
  previewImg.src   = "";
  sampleCards.forEach(c => c.classList.remove("active-sample"));
  dropzone.classList.remove("hidden");
  previewWrap.classList.add("hidden");
  showPlaceholder();
});

// ─── Detect ──────────────────────────────────────────────────

detectBtn.addEventListener("click", () => runDetection());

async function runDetection() {
  if (!selectedFile) return;
  detectBtn.disabled = true;
  showLoading();

  const conf = (parseFloat(confSlider?.value || 45) / 100).toFixed(2);
  const iou  = (parseFloat(iouSlider?.value || 35) / 100).toFixed(2);

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("conf", conf);
    formData.append("iou", iou);

    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    renderResults(data);

  } catch (err) {
    console.error(err);
    showPlaceholder();
    alert(`Detection failed: ${err.message}`);
  } finally {
    detectBtn.disabled = false;
  }
}

// ─── Results renderer ────────────────────────────────────────

function renderResults(data) {
  lastAnnotatedB64 = data.annotated_image;

  // Annotated image
  resultImg.src = `data:image/jpeg;base64,${data.annotated_image}`;

  // Metrics
  detCount.textContent = data.num_detections;
  detSpeed.textContent = data.inference_ms;

  // Detection list
  detectionList.innerHTML = "";
  if (data.detections.length === 0) {
    detectionList.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:16px">No damage detected</p>';
  } else {
    data.detections.forEach((det, i) => {
      const color = CLASS_COLORS[det.class_name] || "#ffffff";
      const item = document.createElement("div");
      item.className = "detection-item";
      item.style.animationDelay = `${i * 50}ms`;
      item.innerHTML = `
        <span class="detection-cls">
          <span class="cls-dot" style="background:${color}"></span>
          ${det.class_name}
        </span>
        <span class="det-conf">${(det.confidence * 100).toFixed(1)}%</span>
      `;
      detectionList.appendChild(item);
    });
  }

  // Show results
  loadingState.classList.add("hidden");
  resultsContent.classList.remove("hidden");
  downloadBtn.classList.remove("hidden");
}

// ─── Download ────────────────────────────────────────────────

downloadBtn.addEventListener("click", () => {
  if (!lastAnnotatedB64) return;
  const link    = document.createElement("a");
  link.href     = `data:image/jpeg;base64,${lastAnnotatedB64}`;
  link.download = "road_damage_detected.jpg";
  link.click();
});

// ─── State helpers ───────────────────────────────────────────

function showPlaceholder() {
  resultsPlaceholder.classList.remove("hidden");
  loadingState.classList.add("hidden");
  resultsContent.classList.add("hidden");
}

function showLoading() {
  resultsPlaceholder.classList.add("hidden");
  loadingState.classList.remove("hidden");
  resultsContent.classList.add("hidden");
}

// ─── Nav active link on scroll ───────────────────────────────

const navPills = document.querySelectorAll(".nav-pill");
const sections = document.querySelectorAll("section[id]");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navPills.forEach((pill) => {
          pill.classList.toggle(
            "active",
            pill.getAttribute("href") === `#${entry.target.id}`
          );
        });
      }
    });
  },
  { threshold: 0.5 }
);

sections.forEach((sec) => observer.observe(sec));
