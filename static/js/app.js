/**
 * Manuscript OCR Workbench — frontend logic.
 *
 * Talks to the FastAPI backend under /api/v1. No build step, no framework —
 * deliberately simple so it's easy to read alongside the API it calls.
 */
(() => {
  "use strict";

  const API = "/api/v1";

  const state = {
    pageId: null,
    segmentation: null,
    ocr: null,
    engines: [],
    selectedEngine: null,
    activeSegmentId: null,
    currentView: "original",
  };

  // ---------------------------------------------------------------- //
  // DOM refs
  // ---------------------------------------------------------------- //
  const el = {
    dropzone: document.getElementById("dropzone"),
    fileInput: document.getElementById("file-input"),
    browseBtn: document.getElementById("browse-btn"),
    workbench: document.getElementById("workbench"),
    newPageBtn: document.getElementById("new-page-btn"),
    runSegBtn: document.getElementById("run-segmentation-btn"),
    runOcrBtn: document.getElementById("run-ocr-btn"),
    engineSelect: document.getElementById("engine-select"),
    pageImage: document.getElementById("page-image"),
    overlaySvg: document.getElementById("overlay-svg"),
    viewportStage: document.getElementById("viewport-stage"),
    segmentHint: document.getElementById("segment-hint"),
    inspectorEmpty: document.getElementById("inspector-empty"),
    segmentList: document.getElementById("segment-list"),
    segmentCount: document.getElementById("segment-count"),
    pageTextFooter: document.getElementById("page-text-footer"),
    pageText: document.getElementById("page-text"),
    exportButtons: document.getElementById("export-buttons"),
    viewToggle: document.getElementById("view-toggle"),
    toast: document.getElementById("toast"),
    stageList: document.getElementById("stage-list"),
  };

  // ---------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------- //
  function showToast(message, kind = "info") {
    el.toast.textContent = message;
    el.toast.className = `toast show ${kind}`;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => el.toast.classList.remove("show"), 3200);
  }

  async function api(path, options = {}) {
    const res = await fetch(`${API}${path}`, options);
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const contentType = res.headers.get("content-type") || "";
    return contentType.includes("application/json") ? res.json() : res;
  }

  function setStageState(stage, mode, label) {
    const li = el.stageList.querySelector(`[data-stage="${stage}"]`);
    if (!li) return;
    li.classList.remove("is-active", "is-done", "is-error");
    if (mode) li.classList.add(`is-${mode}`);
    const stateEl = li.querySelector(`[data-state="${stage}"]`);
    if (stateEl && label) stateEl.textContent = label;
  }

  // ---------------------------------------------------------------- //
  // Stage 1: Upload
  // ---------------------------------------------------------------- //
  el.browseBtn.addEventListener("click", () => el.fileInput.click());
  el.fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleUpload(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("drag-over");
    })
  );
  el.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  });

  async function handleUpload(file) {
    setStageState("upload", "active", "Uploading…");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const result = await api("/pages/upload", { method: "POST", body: formData });
      state.pageId = result.page_id;
      el.pageImage.src = result.original_image_url;

      el.dropzone.classList.add("hidden");
      el.workbench.classList.remove("hidden");

      setStageState("upload", "done", "Complete");
      setStageState("segment", "active", "Ready");
      el.segmentHint.classList.remove("hidden");
      showToast("Page uploaded. Run segmentation when ready.", "success");
    } catch (err) {
      setStageState("upload", "error", "Failed");
      showToast(`Upload failed: ${err.message}`, "error");
    }
  }

  // Exposed so static/js/samples.js can feed a fetched sample image through
  // the exact same path a drag-and-drop upload takes, without this file
  // needing to know anything about samples.
  window.ocrApp = { handleUpload, showToast };

  el.newPageBtn.addEventListener("click", () => {
    state.pageId = null;
    state.segmentation = null;
    state.ocr = null;
    state.activeSegmentId = null;
    el.workbench.classList.add("hidden");
    el.dropzone.classList.remove("hidden");
    el.fileInput.value = "";
    el.overlaySvg.innerHTML = "";
    el.segmentList.innerHTML = "";
    el.segmentList.classList.add("hidden");
    el.inspectorEmpty.classList.remove("hidden");
    el.pageTextFooter.classList.add("hidden");
    el.runOcrBtn.disabled = true;
    ["upload", "segment", "recognize", "export"].forEach((s) => setStageState(s, null, "Not started"));
  });

  // ---------------------------------------------------------------- //
  // Stage 2: Segmentation
  // ---------------------------------------------------------------- //
  el.runSegBtn.addEventListener("click", runSegmentation);

  async function runSegmentation() {
    if (!state.pageId) return;
    setStageState("segment", "active", "Detecting lines…");
    el.runSegBtn.disabled = true;
    try {
      const result = await api(`/pages/${state.pageId}/segmentation/run`, { method: "POST" });
      state.segmentation = result;
      renderSegments(result);
      setStageState("segment", "done", `${result.total_segments} segments`);
      setStageState("recognize", "active", "Ready");
      el.segmentHint.classList.add("hidden");
      el.runOcrBtn.disabled = false;
      showToast(`Found ${result.total_segments} segments.`, "success");
    } catch (err) {
      setStageState("segment", "error", "Failed");
      showToast(`Segmentation failed: ${err.message}`, "error");
    } finally {
      el.runSegBtn.disabled = false;
    }
  }

  function renderSegments(segmentation) {
    el.inspectorEmpty.classList.add("hidden");
    el.segmentList.classList.remove("hidden");
    el.segmentCount.textContent = segmentation.total_segments;

    el.overlaySvg.setAttribute("viewBox", `0 0 ${segmentation.image_width} ${segmentation.image_height}`);
    el.overlaySvg.innerHTML = "";
    segmentation.segments.forEach((seg) => {
      if (!seg.boundary_polygon || !seg.boundary_polygon.length) return;
      const points = seg.boundary_polygon.map((p) => p.join(",")).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      poly.setAttribute("points", points);
      poly.dataset.segmentId = seg.segment_id;
      poly.addEventListener("click", () => setActiveSegment(seg.segment_id));
      el.overlaySvg.appendChild(poly);
    });

    el.segmentList.innerHTML = "";
    segmentation.segments.forEach((seg) => {
      const li = document.createElement("li");
      li.className = "segment-item";
      li.dataset.segmentId = seg.segment_id;
      li.innerHTML = `
        <img class="segment-thumb" src="${seg.crop_url || ""}" alt="Segment ${seg.segment_id}" />
        <div class="segment-body">
          <div class="segment-meta">
            <span class="segment-index">#${seg.segment_id}</span>
          </div>
          <div class="segment-text placeholder">Not recognized yet</div>
        </div>
      `;
      li.addEventListener("click", () => setActiveSegment(seg.segment_id));
      el.segmentList.appendChild(li);
    });
  }

  function setActiveSegment(segmentId) {
    state.activeSegmentId = segmentId;
    el.overlaySvg.querySelectorAll("polygon").forEach((p) => {
      p.classList.toggle("is-active", Number(p.dataset.segmentId) === segmentId);
    });
    el.segmentList.querySelectorAll(".segment-item").forEach((li) => {
      const isActive = Number(li.dataset.segmentId) === segmentId;
      li.classList.toggle("is-active", isActive);
      if (isActive) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }

  // ---------------------------------------------------------------- //
  // View toggle
  // ---------------------------------------------------------------- //
  el.viewToggle.addEventListener("click", (e) => {
    const btn = e.target.closest(".view-toggle-btn");
    if (!btn) return;
    state.currentView = btn.dataset.view;
    el.viewToggle.querySelectorAll(".view-toggle-btn").forEach((b) => b.classList.toggle("active", b === btn));

    if (state.currentView === "overlay" && state.segmentation?.overlay_image_url) {
      el.pageImage.src = state.segmentation.overlay_image_url;
      el.overlaySvg.classList.add("hidden");
    } else {
      api(`/pages/${state.pageId}/segmentation`).then((seg) => {
        el.pageImage.src = seg.original_image_url;
      }).catch(() => {});
      el.overlaySvg.classList.remove("hidden");
    }
  });

  // ---------------------------------------------------------------- //
  // Stage 3: OCR engines + run (streaming)
  // ---------------------------------------------------------------- //
  async function loadEngines() {
    try {
      const result = await api("/pages/ocr/engines");
      state.engines = result.engines;
      el.engineSelect.innerHTML = "";
      result.engines.forEach((eng) => {
        const opt = document.createElement("option");
        opt.value = eng.engine;
        opt.textContent = eng.available ? eng.label : `${eng.label} (unavailable)`;
        opt.disabled = !eng.available;
        el.engineSelect.appendChild(opt);
      });
      const firstAvailable = result.engines.find((e) => e.available);
      if (firstAvailable) {
        el.engineSelect.value = firstAvailable.engine;
        state.selectedEngine = firstAvailable.engine;
      }
      el.engineSelect.disabled = false;
    } catch (err) {
      showToast(`Could not load OCR engines: ${err.message}`, "error");
    }
  }

  el.engineSelect.addEventListener("change", (e) => {
    state.selectedEngine = e.target.value;
  });

  el.runOcrBtn.addEventListener("click", runOCR);

  // Render a single segment result as soon as it arrives
  function renderSegmentResult(r) {
    const li = el.segmentList.querySelector(`[data-segment-id="${r.segment_id}"]`);
    if (!li) return;
    const textEl = li.querySelector(".segment-text");
    textEl.textContent = r.text || "(empty)";
    textEl.classList.remove("placeholder");
    // brief yellow flash so the user sees what just arrived
    textEl.style.transition = "background 0s";
    textEl.style.background = "var(--accent-subtle, #fffbe6)";
    setTimeout(() => {
      textEl.style.transition = "background 0.8s";
      textEl.style.background = "";
    }, 50);

    const metaEl = li.querySelector(".segment-meta");
    const existingConf = metaEl.querySelector(".segment-conf");
    if (existingConf) existingConf.remove();
    if (r.confidence !== null && r.confidence !== undefined) {
      const conf = document.createElement("span");
      conf.className = `segment-conf${r.confidence < 0.7 ? " low" : ""}`;
      conf.textContent = `${Math.round(r.confidence * 100)}%`;
      metaEl.appendChild(conf);
    }
    li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }

  async function runOCR() {
    if (!state.pageId || !state.selectedEngine) return;
    setStageState("recognize", "active", "Recognizing…");
    el.runOcrBtn.disabled = true;

    // reset previous results
    el.segmentList.querySelectorAll(".segment-text").forEach((t) => {
      t.textContent = "Not recognized yet";
      t.classList.add("placeholder");
    });
    el.pageText.textContent = "";
    el.pageTextFooter.classList.add("hidden");

    const url = `${API}/pages/${state.pageId}/ocr/stream?engine=${encodeURIComponent(state.selectedEngine)}`;
    const collectedTexts = {};

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Server error ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by double newline
        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // last item may be incomplete

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;
          let data;
          try { data = JSON.parse(line.slice(5).trim()); } catch (_) { continue; }

          if (data.done) {
            // build page text from collected lines in segment order
            const pageText = Object.keys(collectedTexts)
              .sort((a, b) => Number(a) - Number(b))
              .map((k) => collectedTexts[k])
              .join("\n");
            el.pageText.textContent = pageText || "(no text recognized)";
            el.pageTextFooter.classList.remove("hidden");
            setStageState("recognize", "done", `${data.total_inference_time_ms.toFixed(0)} ms`);
            setStageState("export", "active", "Ready");
            showToast("Recognition complete.", "success");
            el.runOcrBtn.disabled = false;
          } else {
            // show this segment immediately
            renderSegmentResult(data);
            collectedTexts[data.segment_id] = data.text || "";
            const n = Object.keys(collectedTexts).length;
            setStageState("recognize", "active", `${n} lines done…`);
          }
        }
      }
    } catch (err) {
      setStageState("recognize", "error", "Failed");
      showToast(`OCR failed: ${err.message}`, "error");
      el.runOcrBtn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- //
  // Stage 4: Export
  // ---------------------------------------------------------------- //
  el.exportButtons.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-fmt]");
    if (!btn || !state.pageId) return;
    const fmt = btn.dataset.fmt;
    const path = fmt === "alto" ? "alto" : fmt === "page-xml" ? "page-xml" : fmt;
    window.open(`${API}/pages/${state.pageId}/export/${path}?engine=${encodeURIComponent(state.selectedEngine)}`, "_blank");
  });

  // ---------------------------------------------------------------- //
  // Init
  // ---------------------------------------------------------------- //
  loadEngines();
})();