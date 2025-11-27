import os
import xml.etree.ElementTree as ET

RAW_DIR = "raw_data/Houston_TX_USA"
NEED_DIR = "need_data"

os.makedirs(NEED_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.startswith("edgedata") and filename.endswith(".xml"):
        filepath = os.path.join(RAW_DIR, filename)
        tree = ET.parse(filepath)
        root = tree.getroot()

        # New root for filtered XML
        new_root = ET.Element(root.tag)

        count_kept = 0
        for edge in root.findall(".//edge"):
            entered = float(edge.get("entered", 0))
            flow = float(edge.get("flow", 0))
            if entered > 0 or flow > 0:
                new_root.append(edge)
                count_kept += 1

        new_tree = ET.ElementTree(new_root)
        out_path = os.path.join(NEED_DIR, filename)
        new_tree.write(out_path, encoding="UTF-8", xml_declaration=True)
        print(f"{filename}: kept {count_kept} edges")

print("Filtered edge data saved to 'need_data' folder.")