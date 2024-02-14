import godotconfig
from pathlib import Path

for gdextension in Path("project/addons").glob("*/*.gdextension"):
    print(f"Merging gdextension parts into {gdextension}")
    orig_text=godotconfig.read(gdextension)
    op_names=list(gdextension.parent.glob("*.part"))
    other_parts=[godotconfig.read(p) for p in op_names]
    print("Found other parts:",[str(n) for n in op_names])
    
    merged_parts=godotconfig.merge_configurations([orig_text]+other_parts)
    print(godotconfig.get_as_text(merged_parts))
