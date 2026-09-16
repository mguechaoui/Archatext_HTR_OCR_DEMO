/**
 * "Try a sample" — lets a visitor run the pipeline without uploading
 * anything, using manuscript pages already sitting in the repo.
 *
 * Deliberately does not touch the backend. A sample image is fetched as a
 * blob and handed to the exact same handleUpload(file) path a drag-and-drop
 * upload goes through — from the API's point of view this is a normal
 * upload, so /api/v1/pages/upload needs no changes at all.
 *
 * app.js must load before this file (see index.html) and must expose
 * window.ocrApp.handleUpload for this to work.
 */
(() => {
  "use strict";


  const SAMPLES = [
    { file: "sample1.jpg", label: "Sample page 1" },
    { file: "sample2.jpg", label: "Sample page 2" },
    { file: "sample3.jpg", label: "Sample page 3" },
  ];

  const container = document.getElementById("sample-gallery");
  if (!container) return; // markup not present — nothing to wire up

  async function useSample(sample, btn) {
    const original = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Loading…";
    try {
      // static/ is mounted at "/", so static/samples/x.jpg is served at
      // /samples/x.jpg, not static/samples/x.jpg.
      const res = await fetch(`/samples/${sample.file}`);
      if (!res.ok) {
        throw new Error(
          `${sample.file} not found. Add your own sample images to ` +
          `static/samples/ — see static/samples/README.md.`
        );
      }
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: blob.type || "image/jpeg" });

      if (!window.ocrApp || typeof window.ocrApp.handleUpload !== "function") {
        throw new Error("Upload handler not ready yet — try again in a moment.");
      }
      await window.ocrApp.handleUpload(file);
    } catch (err) {
      if (window.ocrApp && typeof window.ocrApp.showToast === "function") {
        window.ocrApp.showToast(err.message, "error");
      } else {
        console.error(err);
        alert(err.message);
      }
    } finally {
      btn.disabled = false;
      btn.textContent = original;
    }
  }

  SAMPLES.forEach((sample) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-ghost sample-btn";
    btn.textContent = sample.label;
    btn.addEventListener("click", () => useSample(sample, btn));
    container.appendChild(btn);
  });
})();