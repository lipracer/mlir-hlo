# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Workspace for MLIR HLO."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

LLVM_COMMIT = "8001dcbd50ce501e75e06886c5e225298e9eea2b"

LLVM_SHA256 = "4f1c75a677aa5df478b3361f5dd3e06555f033826cb0d99be9432a54c838e9e3"

LLVM_BAZEL_TAG = "llvm-project-{commit}".format(commit = LLVM_COMMIT)

http_archive(
    name = "llvm-bazel",
    strip_prefix = "llvm-bazel-{tag}/llvm-bazel".format(tag = LLVM_BAZEL_TAG),
    url = "https://github.com/google/llvm-bazel/archive/{tag}.tar.gz".format(tag = LLVM_BAZEL_TAG),
)

load("@llvm-bazel//:terminfo.bzl", "llvm_terminfo_disable")
load("@llvm-bazel//:zlib.bzl", "llvm_zlib_disable")
load("@llvm-bazel//:configure.bzl", "llvm_configure")

http_archive(
    name = "llvm-project-raw",
    build_file_content = "#empty",
    sha256 = LLVM_SHA256,
    strip_prefix = "llvm-project-{commit}".format(commit = LLVM_COMMIT),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
        "https://github.com/llvm/llvm-project/archive/{commit}.tar.gz".format(commit = LLVM_COMMIT),
    ],
)

llvm_terminfo_disable(
    name = "llvm_terminfo",
)

llvm_zlib_disable(
    name = "llvm_zlib",
)

llvm_configure(
    name = "llvm-project",
    src_path = ".",
    src_workspace = "@llvm-project-raw//:WORKSPACE",
)