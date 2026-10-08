"""Pascal VOC parsing with class-agnostic competition semantics."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VocObject:
    source_class: str
    box: tuple


@dataclass(frozen=True)
class VocSample:
    sample_id: str
    image_path: Path
    xml_path: Path
    width: int
    height: int
    objects: tuple


def find_voc_directories(root):
    """Accept canonical VOC names and the common images/annotations variant."""
    root = Path(root).resolve()
    image_dir = root / "JPEGImages"
    if not image_dir.is_dir():
        image_dir = root / "images"
    annotation_dir = root / "Annotations"
    if not annotation_dir.is_dir():
        annotation_dir = root / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            "VOC data must contain JPEGImages + Annotations "
            f"(or images + annotations): {root}"
        )
    return image_dir, annotation_dir


def find_image(image_dir, sample_id, xml_root=None):
    """Resolve the image named by XML first, then fall back to the sample stem."""
    candidates = []
    if xml_root is not None:
        filename = (xml_root.findtext("filename") or "").strip()
        if filename:
            candidates.append(Path(image_dir) / Path(filename).name)
    candidates.extend(Path(image_dir) / f"{sample_id}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image for XML sample '{sample_id}' was not found in {image_dir}")


def load_voc_sample(xml_path, image_dir, verify_image=True):
    """Load one VOC sample and collapse every object name to a generic target."""
    xml_path = Path(xml_path)
    root = ElementTree.parse(str(xml_path)).getroot()
    sample_id = xml_path.stem
    image_path = find_image(image_dir, sample_id, root)

    xml_width = int(float(root.findtext("size/width") or 0))
    xml_height = int(float(root.findtext("size/height") or 0))
    with Image.open(image_path) as image:
        image_width, image_height = image.size
    if verify_image and (xml_width, xml_height) != (image_width, image_height):
        raise ValueError(
            f"XML/image size mismatch for {sample_id}: "
            f"XML={xml_width}x{xml_height}, image={image_width}x{image_height}"
        )
    width = image_width
    height = image_height

    objects = []
    for index, node in enumerate(root.findall("object"), start=1):
        box_node = node.find("bndbox")
        if box_node is None:
            raise ValueError(f"Missing bndbox in {xml_path}, object {index}")
        x1 = float(box_node.findtext("xmin"))
        y1 = float(box_node.findtext("ymin"))
        x2 = float(box_node.findtext("xmax"))
        y2 = float(box_node.findtext("ymax"))
        x1 = max(0.0, min(float(width), x1))
        y1 = max(0.0, min(float(height), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box in {xml_path}, object {index}: {(x1, y1, x2, y2)}")
        source_class = (node.findtext("name") or "target").strip() or "target"
        objects.append(VocObject(source_class=source_class, box=(x1, y1, x2, y2)))

    return VocSample(
        sample_id=sample_id,
        image_path=image_path,
        xml_path=xml_path,
        width=width,
        height=height,
        objects=tuple(objects),
    )


def read_split_ids(root, split):
    path = Path(root) / "ImageSets" / "Main" / f"{split}.txt"
    if not path.is_file():
        return None
    ids = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip().split()[0] if line.strip() else ""
        if value:
            ids.append(Path(value).stem)
    return ids


def discover_ids(annotation_dir):
    return sorted(path.stem for path in Path(annotation_dir).glob("*.xml"))
