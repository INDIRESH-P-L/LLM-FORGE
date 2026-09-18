import json

files = {
    "/home/sece2026-student12/LegalMindAI/app/static/style.css": [],
    "/home/sece2026-student12/LegalMindAI/app/static/index.html": [],
    "/home/sece2026-student12/LegalMindAI/app/static/app.js": []
}

with open("/home/sece2026-student12/.gemini/antigravity-ide/brain/23ad39c2-7122-49ee-8536-33f7e1c8f0f3/.system_generated/logs/transcript_full.jsonl", "r") as f:
    for line in f:
        step = json.loads(line)
        if step.get("type") == "PLANNER_RESPONSE":
            for tool_call in step.get("tool_calls", []):
                name = tool_call.get("name", "")
                args = tool_call.get("args", {})
                
                target = args.get("TargetFile")
                if target in files:
                    if name == "multi_replace_file_content":
                        for chunk in args.get("ReplacementChunks", []):
                            files[target].append({
                                "TargetContent": chunk["TargetContent"],
                                "ReplacementContent": chunk["ReplacementContent"]
                            })
                    elif name == "replace_file_content":
                        files[target].append({
                            "TargetContent": args["TargetContent"],
                            "ReplacementContent": args["ReplacementContent"]
                        })

for path, edits in files.items():
    if not edits:
        print(f"No edits found for {path}")
        continue
        
    with open(path, "r") as f:
        content = f.read()
    
    # Revert edits in reverse order
    for edit in reversed(edits):
        # We want to replace ReplacementContent with TargetContent
        # Verify first if ReplacementContent exists in content
        if edit["ReplacementContent"] in content:
            content = content.replace(edit["ReplacementContent"], edit["TargetContent"], 1)
        else:
            print(f"Warning: could not find ReplacementContent in {path}")
    
    with open(path, "w") as f:
        f.write(content)
    
    print(f"Reverted {path}")

