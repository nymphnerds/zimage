from __future__ import annotations

import json
import re


DEFAULT_PART_EXTRACTION_GUIDANCE = (
    "Preserve the exact illustration style, proportions, color palette, material details, and scale relationship from the reference image. "
    "Preserve the character's inferred body type and silhouette proportions from the master image, including short, squat, stout, bulky, chibi, or stylized proportions when present. "
    "Do not replace the character with a generic slim adult anatomy template. "
    "Match the same line weight, brush texture, paper texture, color softness, rendering flatness, and level of detail. "
    "Use the same media feel as the source image on a simple plain background. "
    "Do not convert the result into a 3D render, product render, glossy material render, realistic render, toy render, or studio product photo. "
    "Do not include unrelated body parts, heads, mannequins, labels, extra objects, scenery, or shadows."
)


def extract_json_payload(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def normalized_bbox(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return []
    normalized = []
    for item in value:
        try:
            normalized.append(max(0.0, min(1.0, float(item))))
        except Exception:
            return []
    return normalized


def slugify_part_id(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return slug or fallback


def _part_identifier(part: dict) -> str:
    return " ".join(
        str(part.get(key) or "").strip().lower()
        for key in ("id", "display_name", "category", "extraction_prompt")
    )


def part_mentions_eye(part: dict) -> bool:
    return bool(re.search(r"\b(?:eye|eyes|eyeball|eyeballs|iris|pupil)\b", _part_identifier(part)))


def part_is_face_detail(part: dict) -> bool:
    category = (part.get("category") or "").strip().lower()
    identifier = _part_identifier(part)
    if any(word in identifier for word in ("beard", "moustache", "mustache", "facial hair", "hair")):
        return False
    if category == "face_feature":
        return True
    return any(
        word in identifier
        for word in (
            " face ",
            "facial",
            "face detail",
            "facial detail",
            "nose",
            "mouth",
            "lip",
            "eyebrow",
            "brow",
            "ear",
            "cheek",
            "jaw",
        )
    )


def normalize_part_plan(raw_plan: dict, *, max_parts: int = 8) -> dict:
    if not isinstance(raw_plan, dict):
        raise RuntimeError("Planner JSON must be an object with a parts list.")
    raw_parts = raw_plan.get("parts")
    if not isinstance(raw_parts, list):
        raise RuntimeError("Planner JSON did not contain a parts list.")

    parts = []
    seen = set()
    for index, raw_part in enumerate(raw_parts, start=1):
        if not isinstance(raw_part, dict):
            continue
        display_name = str(raw_part.get("display_name") or raw_part.get("name") or "").strip()
        category = str(raw_part.get("category") or "part").strip().lower()
        fallback_id = f"part_{index:02d}"
        part_id = slugify_part_id(raw_part.get("id") or display_name, fallback_id)
        if part_id in seen:
            part_id = f"{part_id}_{index:02d}"
        seen.add(part_id)
        if not display_name:
            display_name = part_id.replace("_", " ").title()
        extraction_prompt = str(raw_part.get("extraction_prompt") or raw_part.get("prompt") or "").strip()
        if not extraction_prompt:
            extraction_prompt = (
                f"Extract only the {display_name}. Remove all unrelated body parts, heads, mannequins, "
                "labels, extra objects, and background elements."
            )
        try:
            priority = int(raw_part.get("priority") or index)
        except Exception:
            priority = index
        parts.append(
            {
                "id": part_id,
                "display_name": display_name,
                "category": category,
                "priority": priority,
                "selected": bool(raw_part.get("selected", True)),
                "symmetry": bool(raw_part.get("symmetry", False)),
                "normalized_bbox": normalized_bbox(raw_part.get("normalized_bbox")),
                "extraction_prompt": extraction_prompt,
            }
        )

    if not parts:
        raise RuntimeError("Planner did not identify any extractable parts.")

    parts.sort(key=lambda item: (item.get("priority", 999), item.get("id", "")))
    return {"parts": parts[: max(1, int(max_parts or 8))]}


def forced_anatomy_base_part(*, priority: int = 1) -> dict:
    return {
        "id": "anatomy_base",
        "display_name": "Anatomy Base",
        "category": "anatomy_base",
        "priority": int(priority or 1),
        "selected": True,
        "symmetry": False,
        "normalized_bbox": [],
        "extraction_prompt": (
            "Extract the anatomy base body only as a clean reusable base mesh reference. "
            "Infer it from the source character's silhouette and costume volume. "
            "Keep the source body type and proportions. "
            "Remove all clothing, hair, beard, accessories, weapons, props, and costume remnants."
        ),
    }


def single_eye_part(*, priority: int = 2, normalized_bbox_value=None, symmetry: bool = False) -> dict:
    return {
        "id": "eyeball",
        "display_name": "Eyeball",
        "category": "face_feature",
        "priority": int(priority or 2),
        "selected": True,
        "symmetry": bool(symmetry),
        "normalized_bbox": normalized_bbox(normalized_bbox_value or []),
        "extraction_prompt": (
            "Create exactly one isolated spherical eyeball asset inferred from the source character. "
            "Show only the eyeball itself: sclera, iris, pupil, cornea highlight, and subtle painted surface detail. "
            "Match the source iris design, pupil treatment, sclera treatment, highlights, color, and rendering style. "
            "Do not include eyelids, eyelashes, skin, brow, eye socket, tear duct, surrounding flesh, makeup, face crop, head, or any anatomical tissue outside the eyeball. "
            "Center the single eyeball on a plain light background as a clean reusable 3D asset reference."
        ),
    }


def ensure_required_part_plan_entries(
    plan: dict,
    *,
    max_parts: int = 8,
    include_eye_part: bool = False,
) -> dict:
    parts = [part for part in list((plan or {}).get("parts") or []) if not part_is_face_detail(part)]
    anatomy_parts = [
        part
        for part in parts
        if (part.get("category") or "").strip().lower() == "anatomy_base"
        or "anatomy" in _part_identifier(part)
        or "base body" in _part_identifier(part)
    ]
    if anatomy_parts:
        best_base = min(anatomy_parts, key=lambda item: int(item.get("priority") or 999))
        forced_base = forced_anatomy_base_part(priority=best_base.get("priority") or 1)
        parts = [part for part in parts if part not in anatomy_parts]
        parts.append(forced_base)
    else:
        parts.append(forced_anatomy_base_part(priority=1))

    if include_eye_part:
        eye_parts = [part for part in parts if part_mentions_eye(part)]
        if eye_parts:
            best_eye = min(eye_parts, key=lambda item: int(item.get("priority") or 999))
            forced_eye = single_eye_part(
                priority=best_eye.get("priority") or 2,
                normalized_bbox_value=best_eye.get("normalized_bbox") or [],
                symmetry=best_eye.get("symmetry", False),
            )
            parts = [part for part in parts if not part_mentions_eye(part)]
        else:
            insert_priority = 2 if any((part.get("category") or "") == "anatomy_base" for part in parts) else 1
            forced_eye = single_eye_part(priority=insert_priority)
        parts.append(forced_eye)
    return normalize_part_plan({"parts": parts}, max_parts=max_parts)


def part_planning_prompt(
    guidance: str,
    *,
    base_include_face: bool = False,
    base_include_eyes: bool = False,
    include_eye_part: bool = False,
) -> str:
    guidance = (guidance or DEFAULT_PART_EXTRACTION_GUIDANCE).strip()
    base_face_rule = (
        "- For anatomy_base, keep facial features on the base body and match the source character's face structure.\n"
        if base_include_face
        else "- For anatomy_base, keep the head feature-neutral with no finished face details, eyes, nose, mouth, or brows.\n"
    )
    base_eyes_rule = ""
    if base_include_face and base_include_eyes:
        base_eyes_rule = (
            "- For anatomy_base, include finished eyes on the base body and match their placement, shape, and stylization to the source.\n"
        )
    elif base_include_face:
        base_eyes_rule = (
            "- For anatomy_base, keep the face but do not include finished eyes, eyeballs, pupils, lashes, or painted eye detail.\n"
        )
    eye_part_rule = (
        "- Include one separate reusable Eyeball part with category face_feature. It must be exactly one isolated spherical eyeball asset only: sclera, iris, pupil, cornea highlight, and painted surface detail. Do not include eyelids, eyelashes, skin, brow, eye socket, tear duct, face crop, head, or surrounding flesh.\n"
        if include_eye_part
        else ""
    )
    return (
        "Look at the provided master character reference image and plan separate asset extractions for a 3D game asset workflow.\n\n"
        "Return JSON only. Do not include markdown fences or commentary.\n\n"
        "Schema:\n"
        "{\n"
        "  \"parts\": [\n"
        "    {\n"
        "      \"id\": \"short_stable_slug\",\n"
        "      \"display_name\": \"Human readable part name\",\n"
        "      \"category\": \"anatomy_base | hair | clothing | armor | accessory | weapon | prop | face_feature\",\n"
        "      \"priority\": 1,\n"
        "      \"normalized_bbox\": [0.0, 0.0, 1.0, 1.0],\n"
        "      \"extraction_prompt\": \"Specific instruction for isolating only this part from the master reference image\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Planning rules:\n"
        "- Include one anatomy_base part first when a body/base mesh reference is visible or inferable.\n"
        "- For anatomy_base, the extraction_prompt must ask for the body/base mesh only, not a dressed character.\n"
        "- For anatomy_base, include body-type hints in the extraction_prompt, such as short, squat, stocky, stout, bulky, tiny, chibi, elderly, broad, or slim when visible or inferable from the silhouette.\n"
        "- For anatomy_base, explicitly remove all hair, beard, cloak, robe, hood, hat, tunic, boots, belts, weapons, props, accessories, fabric, and costume details.\n"
        "- For anatomy_base, request a smooth non-explicit base mesh body reference with no explicit sexual detail and no censor bars, stickers, blur, fabric panels, or added coverings.\n"
        f"{base_face_rule}"
        f"{base_eyes_rule}"
        "- Do not create separate face-detail parts such as nose, mouth, cheeks, ears, brows, or generic face-detail sheets.\n"
        "- Facial structure belongs on anatomy_base, not as a separate extracted part.\n"
        "- Include hair and facial hair as their own separate part, never attached to clothing.\n"
        "- Include each major garment, armor piece, weapon, carried prop, pouch, belt, or accessory that would matter for 3D asset creation.\n"
        f"{eye_part_rule}"
        "- Do not include scenery, ground shadows, background decorations, labels, or duplicate variants.\n"
        "- Keep the list practical and cost-aware: prefer the most important 4 to 8 parts.\n"
        "- Use normalized_bbox values in x_min, y_min, x_max, y_max order, from 0 to 1, estimating the source-image location.\n"
        "- Each extraction_prompt must ask for exactly one isolated target item and must explicitly remove unrelated body parts, heads, mannequins, labels, extra objects, and background elements.\n\n"
        f"Global extraction guidance to incorporate: {guidance}\n\n"
        "Critical output constraints:\n"
        "- Output valid JSON only.\n"
        "- No markdown fences, no prose, no explanations.\n"
        "- No duplicate parts.\n"
        "- No combined multi-item parts.\n"
        "- Put the most important exclusions inside each extraction_prompt at the end."
    )


def part_extraction_prompt(
    part: dict,
    guidance: str,
    style_text: str = "",
    style_label: str = "",
    *,
    base_include_face: bool = False,
    base_include_eyes: bool = False,
) -> str:
    display_name = part.get("display_name") or part.get("id") or "character part"
    extraction_prompt = (part.get("extraction_prompt") or "").strip()
    category = (part.get("category") or "").strip().lower()
    guidance = (guidance or DEFAULT_PART_EXTRACTION_GUIDANCE).strip()
    style_text = (style_text or "").strip()
    style_label = (style_label or "").strip()
    style_block = ""
    if style_text:
        style_block = (
            "\n\nStyle lock:\n"
            "The extracted part must match the master image's exact visual style. "
            f"Selected style preset: {style_label or 'Custom Style'}. "
            f"Apply this style strongly: {style_text.rstrip().rstrip('.')}. "
            "This style instruction is more important than generic asset cleanup. "
            "Do not change the medium, linework, paint handling, or rendering style while isolating the item."
        )
    bbox = part.get("normalized_bbox") or []
    bbox_hint = f"\nEstimated source location normalized bbox: {bbox}." if bbox else ""
    body_type_block = ""
    symmetry_block = ""
    critical_exclusions = [
        "Do not create a parts sheet, grid, lineup, collage, catalog page, or multi-item layout.",
        "Do not include labels, text, scenery, shadows, or unrelated objects.",
        "Do not reinterpret the item as a product render, 3D render, realistic render, or glossy studio object.",
    ]
    identifier = f"{part.get('id', '')} {display_name} {category}".lower()
    if category == "anatomy_base" or "anatomy" in identifier or "base body" in identifier:
        face_rule = (
            "Keep the face on the base body and match the source character's facial structure."
            if base_include_face
            else "Keep the head feature-neutral with no eyes, nose, mouth, brows, beard detail, or finished facial features."
        )
        eyes_rule = ""
        if base_include_face and base_include_eyes:
            eyes_rule = " Include finished eyes on the base body and match their placement, shape, and stylization to the source."
        elif base_include_face:
            eyes_rule = (
                " Keep the full face structure, nose, mouth, cheeks, jaw, and brow shape, but do not include finished eyes. "
                "Leave the eye area blank, unrendered, or closed with no visible eyeballs, irises, pupils, sclera, lashes, or painted eye detail."
            )
        extraction_prompt = (
            f"{extraction_prompt} "
            "This is the body/base mesh target only. Remove every costume, garment, cloak, robe, hood, hat, boot, belt, prop, staff, hair, beard, vine, and accessory. "
            "Output only the inferred smooth non-explicit body silhouette for the same character proportions. "
            f"{face_rule}{eyes_rule}"
        ).strip()
        body_type_block = (
            "\n\nAnatomy base rules:\n"
            "The target is the base body only. Do not extract or redraw clothing. "
            "Infer the base body from the master character's visible silhouette and costume volume. "
            "Preserve the source character's body type, height impression, age impression, and stylized proportions. "
            "If the source character appears short, squat, stout, round, bulky, elderly, chibi, or dwarf-like, the anatomy base must keep those proportions. "
            "Do not output a generic slim young adult, fashion figure, athletic template, or unrelated anatomy chart. "
            "Create a smooth, non-explicit, feature-neutral base mesh body reference suitable for sculpting and shape generation. "
            "No cloak, robe, hood, hat, tunic, boots, belts, gloves, hair, beard, staff, bag, vines, accessories, props, labels, scenery, fabric panels, blur, censor bars, stickers, or added coverings. "
            "Do not add underwear, shorts, modesty cloth, or costume remnants; use a simplified non-explicit surface with no explicit sexual detail. "
            f"{face_rule}{eyes_rule}"
        )
        if base_include_face and not base_include_eyes:
            critical_exclusions.extend(
                [
                    "For the anatomy base, do not show finished eyes.",
                    "No eyeballs, irises, pupils, sclera, eyelashes, eyeliner, or painted eye detail.",
                    "Do not turn the face into a blank mannequin head with no nose or mouth.",
                ]
            )
        elif not base_include_face:
            critical_exclusions.extend(
                [
                    "For the anatomy base, no finished face details.",
                    "No eyes, nose, mouth, brows, beard detail, or portrait-style facial rendering.",
                ]
            )
    if category in {"clothing", "armor"} or any(
        word in identifier for word in ("boot", "glove", "sleeve", "cloak", "robe", "hood", "pauldron")
    ):
        critical_exclusions.append(
            "Do not include a head, face, hair, full body, mannequin, or hands unless structurally necessary for readability."
        )
    if not part_mentions_eye(part):
        critical_exclusions.extend(
            [
                "Do not add isolated eyes or eyeballs.",
                "Do not add floating facial parts.",
            ]
        )
        if not (category == "anatomy_base" and base_include_face and base_include_eyes):
            critical_exclusions.append("Do not include finished eyes unless this specific part explicitly requires them.")
    if category == "hair" or any(word in identifier for word in ("hair", "beard", "moustache", "mustache", "brow")):
        critical_exclusions.extend(
            [
                "Do not return an eyeball, eye crop, or face crop.",
                "Do not turn this request into a portrait or facial feature extraction.",
            ]
        )
    if part_mentions_eye(part):
        extraction_prompt = (
            "Create exactly one isolated spherical eyeball asset from the source character. "
            "Show only the eyeball itself: sclera, iris, pupil, cornea highlight, and subtle painted surface detail. "
            "Match the exact iris design, pupil treatment, sclera treatment, highlights, color, and rendering style from the source. "
            "Center the single eyeball on a plain light background as a clean reusable 3D asset reference."
        )
        critical_exclusions.extend(
            [
                "Show one single spherical eyeball only.",
                "Do not include both eyes.",
                "Do not include eyelids, eyelashes, skin, brow, eye socket, tear duct, surrounding flesh, makeup, or any anatomical tissue outside the eyeball.",
                "Do not include a full face, a full head, hair, brows, nose, cheeks, ears, or forehead.",
                "Do not return an eye-region crop; return one clean isolated eyeball asset.",
            ]
        )
    if bool(part.get("symmetry", False)):
        symmetry_block = (
            "\n\nSymmetry lock:\n"
            "The output must be perfectly left-right symmetrical and front-readable for easy refinement. "
            "Match silhouette, seams, trim, folds, color placement, wear, and detail density evenly on both sides. "
            "Do not introduce asymmetrical damage, drift, missing details, uneven hems, or side-to-side design differences."
        )
        if category in {"clothing", "armor"} or any(
            word in identifier for word in ("boot", "glove", "sleeve", "cloak", "robe", "hood", "pauldron")
        ):
            symmetry_block += " Treat this wearable as a symmetry-critical asset."
    return (
        "Using the provided master character reference image, create one clean isolated asset reference image.\n\n"
        f"Target part: {display_name}\n\n"
        f"Extraction instruction: {extraction_prompt}\n\n"
        f"Global guidance: {guidance}{style_block}{body_type_block}{symmetry_block}{bbox_hint}\n\n"
        "Output rules:\n"
        "- Show exactly one centered target item.\n"
        "- Preserve the original design language, media style, material details, color palette, and scale relationship from the reference.\n"
        "- Keep the same lighting and media feel as the source image; use a simple plain background only to isolate the item.\n"
        "\nCritical exclusions:\n- "
        + "\n- ".join(critical_exclusions)
    )
