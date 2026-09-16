"""
Export service — turns a completed (segmentation + OCR) page result into
downloadable files in several standard formats.

ALTO XML and PAGE XML are the two standard interchange formats used in the
document-recognition / digital-humanities world; producing them (rather
than just TXT/JSON) is what makes results usable in other tools
(Transkribus, eScriptorium, etc.).
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

from app.config import settings
from app.core.exceptions import PageNotFoundError, SegmentationNotRunError
from app.models import OCREngine
from app.services.page_store import PageStore

logger = logging.getLogger(__name__)


class ExportService:
    def __init__(self, page_store: PageStore):
        self.page_store = page_store

    def _get_completed_page(self, page_id: str, engine_id: OCREngine | None = None):
        state = self.page_store.get(page_id)
        if state is None:
            raise PageNotFoundError(f"No page found for page_id={page_id}")
        if state.segmentation is None:
            raise SegmentationNotRunError(f"Segmentation not run for page_id={page_id}")

        engine_id = engine_id or state.last_ocr_engine
        ocr_result = state.ocr_results.get(engine_id) if engine_id else None
        return state.segmentation, ocr_result

    # ------------------------------------------------------------------ #
    # TXT
    # ------------------------------------------------------------------ #
    def export_txt(self, page_id: str, engine_id: OCREngine | None = None) -> Path:
        _, ocr_result = self._get_completed_page(page_id, engine_id)
        text = ocr_result.page_text if ocr_result else ""
        out_path = settings.output_dir / "txt" / f"{page_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------ #
    # JSON
    # ------------------------------------------------------------------ #
    def export_json(self, page_id: str, engine_id: OCREngine | None = None) -> Path:
        segmentation, ocr_result = self._get_completed_page(page_id, engine_id)

        ocr_by_segment = {}
        if ocr_result:
            ocr_by_segment = {r.segment_id: r for r in ocr_result.segment_results}

        payload = {
            "page_id": page_id,
            "image": {
                "path": segmentation.source_image_path,
                "width": segmentation.image_width,
                "height": segmentation.image_height,
            },
            "engine": ocr_result.engine.value if ocr_result else None,
            "page_text": ocr_result.page_text if ocr_result else None,
            "segments": [
                {
                    "segment_id": seg.segment_id,
                    "order": seg.order,
                    "bounding_box": seg.bounding_box.as_tuple() if seg.bounding_box else None,
                    "baseline": seg.baseline,
                    "boundary_polygon": seg.boundary_polygon,
                    "text": ocr_by_segment[seg.segment_id].text if seg.segment_id in ocr_by_segment else None,
                    "confidence": ocr_by_segment[seg.segment_id].confidence
                    if seg.segment_id in ocr_by_segment
                    else None,
                }
                for seg in segmentation.segments
            ],
        }

        out_path = settings.output_dir / "json" / f"{page_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #
    def export_csv(self, page_id: str, engine_id: OCREngine | None = None) -> Path:
        segmentation, ocr_result = self._get_completed_page(page_id, engine_id)
        ocr_by_segment = {}
        if ocr_result:
            ocr_by_segment = {r.segment_id: r for r in ocr_result.segment_results}

        out_path = settings.output_dir / "csv" / f"{page_id}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["segment_id", "order", "x0", "y0", "x1", "y1", "text", "confidence"])
            for seg in segmentation.segments:
                bbox = seg.bounding_box.as_tuple() if seg.bounding_box else ("", "", "", "")
                ocr = ocr_by_segment.get(seg.segment_id)
                writer.writerow(
                    [seg.segment_id, seg.order, *bbox, ocr.text if ocr else "", ocr.confidence if ocr else ""]
                )
        return out_path

    # ------------------------------------------------------------------ #
    # ALTO XML
    # ------------------------------------------------------------------ #
    def export_alto_xml(self, page_id: str, engine_id: OCREngine | None = None) -> Path:
        segmentation, ocr_result = self._get_completed_page(page_id, engine_id)
        ocr_by_segment = {}
        if ocr_result:
            ocr_by_segment = {r.segment_id: r for r in ocr_result.segment_results}

        ns = "http://www.loc.gov/standards/alto/ns-v4#"
        root = ET.Element("alto", xmlns=ns)
        description = ET.SubElement(root, "Description")
        ET.SubElement(description, "MeasurementUnit").text = "pixel"
        source = ET.SubElement(description, "sourceImageInformation")
        ET.SubElement(source, "fileName").text = Path(segmentation.source_image_path).name

        layout = ET.SubElement(root, "Layout")
        page_el = ET.SubElement(
            layout,
            "Page",
            ID=f"page_{page_id}",
            WIDTH=str(segmentation.image_width),
            HEIGHT=str(segmentation.image_height),
        )
        print_space = ET.SubElement(page_el, "PrintSpace")

        for seg in segmentation.segments:
            bbox = seg.bounding_box
            line = ET.SubElement(
                print_space,
                "TextLine",
                ID=f"line_{seg.segment_id}",
                HPOS=str(bbox.x0) if bbox else "0",
                VPOS=str(bbox.y0) if bbox else "0",
                WIDTH=str(bbox.width) if bbox else "0",
                HEIGHT=str(bbox.height) if bbox else "0",
            )
            ocr = ocr_by_segment.get(seg.segment_id)
            string_el = ET.SubElement(
                line,
                "String",
                CONTENT=ocr.text if ocr else "",
            )
            if ocr and ocr.confidence is not None:
                string_el.set("WC", f"{ocr.confidence:.4f}")

        out_path = settings.output_dir / "alto" / f"{page_id}.xml"
        out_path.write_text(self._pretty_xml(root), encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------ #
    # PAGE XML
    # ------------------------------------------------------------------ #
    def export_page_xml(self, page_id: str, engine_id: OCREngine | None = None) -> Path:
        segmentation, ocr_result = self._get_completed_page(page_id, engine_id)
        ocr_by_segment = {}
        if ocr_result:
            ocr_by_segment = {r.segment_id: r for r in ocr_result.segment_results}

        ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
        root = ET.Element("PcGts", xmlns=ns)
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "Created").text = datetime.now(UTC).isoformat()

        page_el = ET.SubElement(
            root,
            "Page",
            imageFilename=Path(segmentation.source_image_path).name,
            imageWidth=str(segmentation.image_width),
            imageHeight=str(segmentation.image_height),
        )
        region = ET.SubElement(page_el, "TextRegion", id="region_main")

        for seg in segmentation.segments:
            line_el = ET.SubElement(region, "TextLine", id=f"line_{seg.segment_id}")
            if seg.boundary_polygon:
                coords_pts = " ".join(f"{int(x)},{int(y)}" for x, y in seg.boundary_polygon)
                ET.SubElement(line_el, "Coords", points=coords_pts)
            if seg.baseline:
                baseline_pts = " ".join(f"{int(x)},{int(y)}" for x, y in seg.baseline)
                ET.SubElement(line_el, "Baseline", points=baseline_pts)

            ocr = ocr_by_segment.get(seg.segment_id)
            text_equiv = ET.SubElement(line_el, "TextEquiv")
            if ocr and ocr.confidence is not None:
                text_equiv.set("conf", f"{ocr.confidence:.4f}")
            ET.SubElement(text_equiv, "Unicode").text = ocr.text if ocr else ""

        out_path = settings.output_dir / "page_xml" / f"{page_id}.xml"
        out_path.write_text(self._pretty_xml(root), encoding="utf-8")
        return out_path

    @staticmethod
    def _pretty_xml(root: ET.Element) -> str:
        rough = ET.tostring(root, encoding="utf-8")
        return minidom.parseString(rough).toprettyxml(indent="  ")
