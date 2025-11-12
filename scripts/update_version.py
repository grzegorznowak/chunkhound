#!/usr/bin/env uv run python3
"""
Create a new ChunkHound version tag.

With dynamic versioning via hatch-vcs, versions are automatically
derived from git tags. This script simplifies creating properly formatted tags.

Usage:
    python scripts/update_version.py 4.1.0
    python scripts/update_version.py --bump major|minor|patch
"""

import re
import subprocess
import sys
from pathlib import Path


def get_current_version():
    """Get the current version from the latest git tag."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        tag = result.stdout.strip()
        # Remove 'v' prefix if present
        return tag.lstrip("v")
    except subprocess.CalledProcessError:
        return "0.0.0"


def parse_version(version_str):
    """Parse version string into (major, minor, patch) tuple."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return tuple(map(int, match.groups()))


def bump_version(current, bump_type):
    """Bump version based on type (major, minor, patch)."""
    major, minor, patch = parse_version(current)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def validate_version(version):
    """Validate version format."""
    pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
    if not re.match(pattern, version):
        raise ValueError(
            f"Invalid version format: {version}. Expected: X.Y.Z or X.Y.Z-suffix"
        )


def create_tag(version):
    """Create and push git tag."""
    tag = f"v{version}"

    # Check if tag already exists
    result = subprocess.run(
        ["git", "tag", "-l", tag],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(f"Error: Tag {tag} already exists")
        sys.exit(1)

    # Create annotated tag
    subprocess.run(
        ["git", "tag", "-a", tag, "-m", f"Release {version}"],
        check=True,
    )
    print(f"Created tag {tag}")

    return tag


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <version>", file=sys.stderr)
        print(f"       {sys.argv[0]} --bump <major|minor|patch>", file=sys.stderr)
        print(f"Examples:", file=sys.stderr)
        print(f"  {sys.argv[0]} 4.1.0", file=sys.stderr)
        print(f"  {sys.argv[0]} --bump minor", file=sys.stderr)
        sys.exit(1)

    try:
        # Handle --bump flag
        if sys.argv[1] == "--bump":
            if len(sys.argv) != 3 or sys.argv[2] not in ("major", "minor", "patch"):
                print("Error: --bump requires major, minor, or patch", file=sys.stderr)
                sys.exit(1)

            current = get_current_version()
            version = bump_version(current, sys.argv[2])
            print(f"Bumping version from {current} to {version}")
        else:
            version = sys.argv[1]
            validate_version(version)
            print(f"Creating version {version}")

        tag = create_tag(version)

        print(f"\nSuccessfully created version {version}")
        print(f"\nNext steps:")
        print(f"1. Push the tag: git push origin {tag}")
        print(f"2. Build will automatically use version from git tag")
        print(f"\nNote: Version is now managed via git tags (hatch-vcs)")
        print(f"      No need to manually update version files!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
