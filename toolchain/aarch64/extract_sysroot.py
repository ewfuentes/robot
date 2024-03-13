#! /usr/bin/env python3

# Note that this is not a bazel target
import subprocess
import tarfile
import re
import tempfile
import os
from pathlib import Path


def build_docker_container():
    label = "jetson_sysroot:latest"
    DOCKERFILE_CONTENT = """
    FROM nvcr.io/nvidia/l4t-base:35.4.1

    RUN apt-get update && apt-get install libblas-dev liblapack-dev && rm -rf /var/lib/apt/lists/*
    """
    subprocess.run(["docker", "build",  "-t", label, '-'], input=DOCKERFILE_CONTENT, text=True)
    return label


def export_docker_filesystem(tag):
    # Launch a container
    result = subprocess.run(["docker", "run", "-dt", tag], capture_output=True, text=True)
    container_id = result.stdout.strip()

    TAR_FILE = "/tmp/jetson.tar"
    with open(TAR_FILE, 'wb') as file_out:
        subprocess.run(["docker", "export", container_id], stdout=file_out)

    # Kill the container
    subprocess.run(["docker", "kill", container_id])
    return TAR_FILE


def clean_up_tar_file(file_path):
    paths_to_exclude = [
        "^etc/(?!alternatives).*",
        "^run/.*",
        "^usr/sbin/.*",
        "^usr/share.*",
        "^var/.*",
        "^usr/include/X11/.*",
        "^usr/bin/X11.*",
        "^usr/lib/python.*",
        "^usr/lib/systemd.*",
        "^usr/lib/udev.*",
        "^usr/lib/apt.*",
        "^usr/lib/aarch64-linux-gnu/perl.*",
        "^usr/lib/aarch64-linux-gnu/gtk.*",
        "^usr/lib/aarch64-linux-gnu/gstreamer.*",
        "^usr/lib/aarch64-linux-gnu/gdk.*",
        "^usr/lib/aarch64-linux-gnu/libLLVM.*",
        "^usr/lib/aarch64-linux-gnu/libicudata.*",
        "^usr/lib/aarch64-linux-gnu/dri/.*",
        "^usr/lib/aarch64-linux-gnu/libcodec.*",
        "^usr/lib/aarch64-linux-gnu/libavcodec.*",
        "^usr/lib/aarch64-linux-gnu/libgtk.*",
        "^usr/lib/aarch64-linux-gnu/librsvg.*",
        "^usr/lib/python.*",
    ]

    f = tarfile.TarFile(file_path, mode='r')
    remaining = []

    size_total = 0
    total = 0
    size_kept = 0
    kept = 0
    for member in f:
        total += 1
        size_total += member.size
        should_keep = True
        # print('to test:', member.path)
        for pattern in paths_to_exclude:
            match = re.match(pattern, member.path)
            # print('on pattern:', pattern, match)
            if match is not None:
                should_keep = False
                break

        if should_keep:
            kept += 1
            size_kept += member.size
            remaining.append(member)
    
    for member in remaining:
        print(member, member.size / (1024 ** 2))

    print(f" Files kept / total: {kept} / {total}")
    print(f" Megabytes kept / total: {size_kept / 1024 / 1024} / {size_total / 1024 / 1024}")
    CLEANED_TAR_FILE = "/tmp/cleaned_jetson.tar"
    with tempfile.TemporaryDirectory() as tempdir:
        f.extractall(path=tempdir, members=remaining)
        f.close()

        # make some symlinks
        for obj in ["crt1.o", "crti.o", "crtn.o"]:
            src_path = os.path.join(tempdir, f'lib/aarch64-linux-gnu/{obj}')
            target_path = os.path.join(tempdir, f'lib/{obj}')

            target_dir = os.path.dirname(target_path)
            source_dir = os.path.dirname(src_path)
            rel_src = os.path.relpath(src_path, target_dir)
            print(f'src: {src_path} target: {target_path} target_dir: {target_dir} rel_src: {rel_src}')
            os.symlink(rel_src, target_path)

        # edit /lib/aarch64-linux-gnu/libc.so
        libc_path = Path(tempdir) / 'lib/aarch64-linux-gnu/libc.so'
        contents = libc_path.read_text()
        contents = contents.replace('/usr/lib/aarch64-linux-gnu/', '')
        contents = contents.replace('/lib/aarch64-linux-gnu/', '')
        libc_path.write_text(contents)

        cleaned = tarfile.TarFile(CLEANED_TAR_FILE, mode='w')
        cleaned.add(tempdir, arcname='/')
        cleaned.close()
    return CLEANED_TAR_FILE



def main():
    # Build the docker container
    print('Building Docker Container')
    tag = build_docker_container()

    # Export the filesystem as a tar file
    print('Exporting Docker Container')
    file_path = export_docker_filesystem(tag)

    # Remove unneeded files
    print('Removing Unneeded files')
    cleaned_file_path = clean_up_tar_file(file_path)
    print(cleaned_file_path)


if __name__ == "__main__":
    main()

