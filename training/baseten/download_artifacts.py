"""Download a Baseten checkpoint-files manifest; no simulation or training."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
from urllib.parse import urlsplit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",help="JSON path printed by baseten train checkpoint files")
    parser.add_argument("--output",default="downloaded")
    args=parser.parse_args()
    raw=Path(args.manifest).read_text()
    try:
        manifest=json.loads(raw)
    except json.JSONDecodeError:
        manifest=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(manifest,list):
        entries=manifest
    elif "url" in manifest:
        entries=[manifest]
    else:
        entries=manifest.get("checkpoint_artifacts",manifest.get("files",manifest.get("presigned_urls")))
    if not isinstance(entries,list) or not entries:
        raise ValueError("Manifest contains no checkpoint artifacts; wait for sync and regenerate it")
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        relative=PurePosixPath(entry["relative_file_name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid artifact path")
        rank=int(entry.get("node_rank",0))
        target=(root/f"rank-{rank}"/str(relative)).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Artifact path escapes output directory")
        if urlsplit(entry["url"]).scheme!="https":
            raise ValueError("Checkpoint download requires HTTPS")
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix(target.suffix+".download")
        try:
            with urllib.request.urlopen(entry["url"],timeout=120) as response,partial.open("wb") as output:
                shutil.copyfileobj(response,output)
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            # Do not print expiring signed URLs or their query credentials.
            raise RuntimeError(f"Download failed for {relative}; regenerate the manifest if its URLs expired") from None
        print(target.relative_to(root))
    print(f"Downloaded {len(entries)} files to {root}")


if __name__=="__main__":
    main()
