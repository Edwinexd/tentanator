#!/usr/bin/env python3
"""
Workspace management for Tentanator - organize sessions, models, and data by workspace.

A workspace is a self-contained folder that includes:
- Sessions (.tentanator_sessions/)
- Models (models.json)
- Training data (training_data/)
- Graded exams (graded_exams/, graded_exams_out/)
- Global banks (global_bank/, global_banks_embeddings/)
- Backups (backups/)
- Input exams (exams/, exams_in/, exams_in_raw/)

Usage:
    python workspace.py list                    # List all workspaces
    python workspace.py create <name>           # Create a new workspace
    python workspace.py load <name>             # Load a workspace (move files from workspace to root)
    python workspace.py unload [name]           # Unload current workspace (move files to workspace)
    python workspace.py delete <name>           # Delete a workspace
    python workspace.py current                 # Show current workspace
"""

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional
from aioconsole import ainput


# Workspace configuration
WORKSPACES_DIR = Path("workspaces")
CURRENT_WORKSPACE_FILE = Path(".current_workspace")

# Directories and files that belong to a workspace
WORKSPACE_ITEMS = [
    ".tentanator_sessions",
    "models.json",
    "training_data",
    "graded_exams",
    "graded_exams_out",
    "global_bank",
    "global_banks_embeddings",
    "backups",
    "exams",
    "exams_in",
    "exams_in_raw"
]


def get_current_workspace() -> Optional[str]:
    """Get the name of the currently loaded workspace"""
    if CURRENT_WORKSPACE_FILE.exists():
        try:
            return CURRENT_WORKSPACE_FILE.read_text(encoding='utf-8').strip()
        except OSError:
            pass
    return None


def set_current_workspace(name: Optional[str]) -> None:
    """Set or clear the current workspace marker"""
    if name:
        CURRENT_WORKSPACE_FILE.write_text(name, encoding='utf-8')
    else:
        if CURRENT_WORKSPACE_FILE.exists():
            CURRENT_WORKSPACE_FILE.unlink()


def list_workspaces() -> List[Dict[str, any]]:
    """List all available workspaces with metadata"""
    if not WORKSPACES_DIR.exists():
        return []

    workspaces = []
    for workspace_dir in WORKSPACES_DIR.iterdir():
        if not workspace_dir.is_dir():
            continue

        metadata = {
            "name": workspace_dir.name,
            "path": workspace_dir,
            "sessions": 0,
            "has_models": False,
            "has_training_data": False
        }

        # Count sessions
        sessions_dir = workspace_dir / ".tentanator_sessions"
        if sessions_dir.exists():
            session_files = [
                f for f in sessions_dir.glob("*.json")
                if ".cache.json" not in f.name
            ]
            metadata["sessions"] = len(session_files)

        # Check for models
        models_file = workspace_dir / "models.json"
        if models_file.exists():
            metadata["has_models"] = True

        # Check for training data
        training_dir = workspace_dir / "training_data"
        if training_dir.exists() and any(training_dir.glob("*.jsonl")):
            metadata["has_training_data"] = True

        workspaces.append(metadata)

    return sorted(workspaces, key=lambda x: x["name"])


def create_workspace(name: str) -> bool:
    """Create a new workspace"""
    # Sanitize workspace name
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

    workspace_path = WORKSPACES_DIR / safe_name
    if workspace_path.exists():
        print(f"❌ Workspace '{safe_name}' already exists")
        return False

    # Create workspace directory
    WORKSPACES_DIR.mkdir(exist_ok=True)
    workspace_path.mkdir()

    # Create subdirectories
    (workspace_path / ".tentanator_sessions").mkdir()
    (workspace_path / "training_data").mkdir()
    (workspace_path / "graded_exams").mkdir()
    (workspace_path / "graded_exams_out").mkdir()
    (workspace_path / "backups").mkdir()
    (workspace_path / "exams").mkdir()
    (workspace_path / "exams_in").mkdir()
    (workspace_path / "exams_in_raw").mkdir()

    # Create empty models.json
    with open(workspace_path / "models.json", 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=2)

    print(f"✓ Created workspace: {safe_name}")
    print(f"  Location: {workspace_path}")
    return True


def check_active_items() -> List[str]:
    """Check which workspace items exist in the root directory"""
    active_items = []
    for item in WORKSPACE_ITEMS:
        item_path = Path(item)
        if item_path.exists():
            active_items.append(item)
    return active_items


def load_workspace(name: str) -> bool:
    """Load a workspace (move files from workspace to root)"""
    workspace_path = WORKSPACES_DIR / name
    if not workspace_path.exists():
        print(f"❌ Workspace '{name}' does not exist")
        return False

    # Check if there's already a loaded workspace
    current = get_current_workspace()
    if current:
        print(f"❌ Workspace '{current}' is currently loaded")
        print(f"   Please unload it first with: python workspace.py unload")
        return False

    # Check if there are active items in root
    active_items = check_active_items()
    if active_items:
        print("❌ Cannot load workspace - the following items exist in root:")
        for item in active_items:
            print(f"   - {item}")
        print("\nPlease move or delete these items first, or unload the current workspace.")
        return False

    print(f"📦 Loading workspace: {name}")

    # Move items from workspace to root
    moved_items = []
    for item in WORKSPACE_ITEMS:
        src = workspace_path / item
        dst = Path(item)

        if src.exists():
            try:
                shutil.move(str(src), str(dst))
                moved_items.append(item)
                print(f"  ✓ Moved {item}")
            except (OSError, shutil.Error) as e:
                print(f"  ⚠️  Failed to move {item}: {e}")
                # Rollback - move back items that were already moved
                print("\n⚠️  Rolling back changes...")
                for moved_item in moved_items:
                    try:
                        shutil.move(str(Path(moved_item)), str(workspace_path / moved_item))
                    except (OSError, shutil.Error):
                        pass
                return False

    # Mark this workspace as current
    set_current_workspace(name)

    print(f"\n✅ Workspace '{name}' loaded successfully")
    return True


def unload_workspace(name: Optional[str] = None) -> bool:
    """Unload current workspace (move files from root to workspace)"""
    current = get_current_workspace()

    # Determine which workspace to unload to
    if name:
        # User specified a workspace name
        target_workspace = name
        if current and current != name:
            print(f"⚠️  Warning: Currently loaded workspace is '{current}', but unloading to '{name}'")
    else:
        # Use current workspace
        if not current:
            print("❌ No workspace is currently loaded")
            print("   Specify a workspace name to unload to: python workspace.py unload <name>")
            return False
        target_workspace = current

    workspace_path = WORKSPACES_DIR / target_workspace

    # Create workspace if it doesn't exist
    if not workspace_path.exists():
        print(f"📦 Creating workspace: {target_workspace}")
        create_workspace(target_workspace)

    print(f"📤 Unloading to workspace: {target_workspace}")

    # Check what items exist in root
    active_items = check_active_items()
    if not active_items:
        print("⚠️  No workspace items found in root directory")
        set_current_workspace(None)
        return True

    # Move items from root to workspace
    moved_items = []
    for item in WORKSPACE_ITEMS:
        src = Path(item)
        dst = workspace_path / item

        if src.exists():
            # If destination exists, remove it first (we're replacing with current state)
            if dst.exists():
                try:
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                except OSError as e:
                    print(f"  ⚠️  Failed to remove existing {item} in workspace: {e}")
                    continue

            try:
                shutil.move(str(src), str(dst))
                moved_items.append(item)
                print(f"  ✓ Moved {item}")
            except (OSError, shutil.Error) as e:
                print(f"  ⚠️  Failed to move {item}: {e}")

    # Clear current workspace marker
    set_current_workspace(None)

    print(f"\n✅ Workspace unloaded to '{target_workspace}'")
    return True


async def delete_workspace(name: str, force: bool = False) -> bool:
    """Delete a workspace"""
    workspace_path = WORKSPACES_DIR / name
    if not workspace_path.exists():
        print(f"❌ Workspace '{name}' does not exist")
        return False

    # Check if it's the current workspace
    current = get_current_workspace()
    if current == name:
        print(f"❌ Cannot delete workspace '{name}' - it is currently loaded")
        print(f"   Please unload it first with: python workspace.py unload")
        return False

    # Confirm deletion
    if not force:
        print(f"⚠️  WARNING: This will permanently delete workspace '{name}'")
        print(f"   Location: {workspace_path}")
        response = (await ainput("   Type 'yes' to confirm: ")).strip().lower()
        if response != 'yes':
            print("Deletion cancelled")
            return False

    # Delete the workspace
    try:
        shutil.rmtree(workspace_path)
        print(f"✓ Deleted workspace: {name}")
        return True
    except OSError as e:
        print(f"❌ Failed to delete workspace: {e}")
        return False


def show_current_workspace() -> None:
    """Show the current workspace status"""
    current = get_current_workspace()

    if current:
        print(f"📍 Current workspace: {current}")
        workspace_path = WORKSPACES_DIR / current
        if workspace_path.exists():
            print(f"   Location: {workspace_path}")
        else:
            print(f"   ⚠️  Workspace directory not found at: {workspace_path}")
    else:
        print("📍 No workspace currently loaded")

    # Show active items in root
    active_items = check_active_items()
    if active_items:
        print("\n📂 Active items in root directory:")
        for item in active_items:
            item_path = Path(item)
            if item_path.is_dir():
                # Count items in directory
                if item == ".tentanator_sessions":
                    session_count = len([
                        f for f in item_path.glob("*.json")
                        if ".cache.json" not in f.name
                    ])
                    print(f"   • {item}/ ({session_count} sessions)")
                else:
                    try:
                        count = sum(1 for _ in item_path.rglob("*") if _.is_file())
                        print(f"   • {item}/ ({count} files)")
                    except OSError:
                        print(f"   • {item}/")
            else:
                print(f"   • {item}")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Manage Tentanator workspaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    subparsers.add_parser("list", help="List all workspaces")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new workspace")
    create_parser.add_argument("name", help="Name of the workspace to create")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load a workspace")
    load_parser.add_argument("name", help="Name of the workspace to load")

    # Unload command
    unload_parser = subparsers.add_parser("unload", help="Unload current workspace")
    unload_parser.add_argument(
        "name",
        nargs="?",
        help="Name of the workspace to unload to (defaults to current)"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a workspace")
    delete_parser.add_argument("name", help="Name of the workspace to delete")
    delete_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt"
    )

    # Current command
    subparsers.add_parser("current", help="Show current workspace")

    args = parser.parse_args()

    # Execute command
    if args.command == "list":
        workspaces = list_workspaces()
        current = get_current_workspace()

        if not workspaces:
            print("No workspaces found")
            print("\nCreate a new workspace with: python workspace.py create <name>")
            return

        print(f"📁 Available workspaces ({len(workspaces)}):\n")
        for ws in workspaces:
            marker = "→" if ws["name"] == current else " "
            print(f"{marker} {ws['name']}")
            print(f"    Sessions: {ws['sessions']}")
            print(f"    Models: {'Yes' if ws['has_models'] else 'No'}")
            print(f"    Training data: {'Yes' if ws['has_training_data'] else 'No'}")
            print()

        if current:
            print(f"Currently loaded: {current}")

    elif args.command == "create":
        create_workspace(args.name)

    elif args.command == "load":
        load_workspace(args.name)

    elif args.command == "unload":
        unload_workspace(args.name)

    elif args.command == "delete":
        await delete_workspace(args.name, args.force)

    elif args.command == "current":
        show_current_workspace()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
