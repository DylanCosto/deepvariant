/*
 * Copyright 2024 Google LLC.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "third_party/nucleus/io/example_writer.h"
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/nucleus/io/tfrecord_reader.h"
#include "tensorflow/core/platform/test.h"

namespace nucleus {

TEST(ExampleWriterTest, TFErrorFilename) {
  auto output_record = std::string("test_output");
  EXPECT_DEATH(ExampleWriter("/tmp/out.csv"),
  "Unsupported file extension");
}

TEST(ExampleWriterTest, TFUpperCase) {
  auto output_record = std::string("test_output");
  ExampleWriter writer = ExampleWriter("/tmp/out.TFRECORD.GZ");
  EXPECT_TRUE(writer.Add(output_record));
}


TEST(ExampleWriterTest, NoParentDir) {
  // Tests for when no directory needs to be created.
  auto output_record = std::string("test_output");
  std::filesystem::current_path("/tmp");
  ExampleWriter writer = ExampleWriter("out.TFRECORD.GZ");
  EXPECT_TRUE(writer.Add(output_record));
}


TEST(ExampleWriterTest, TFWriterTest) {
  auto output_record = std::string("test_output");
  ExampleWriter writer = ExampleWriter("/tmp/out.tfrecord.gz");
  EXPECT_TRUE(writer.Add(output_record));
}

TEST(ExampleWriterTest, InvalidCompressionLevel) {
  for (int level : {-2, 10, 128}) {
    ExampleWriter writer(
        ::testing::TempDir() + "invalid-compression.tfrecord.gz",
        ExampleFormat::kAuto, level);
    EXPECT_FALSE(writer.status().ok());
    EXPECT_FALSE(writer.Add("record"));
  }
}

TEST(ExampleWriterTest, CompressionLevelsPreserveRecords) {
  const std::vector<std::string> records = {"", std::string(4096, 'A'),
                                            std::string("binary\0record", 13),
                                            std::string(65536, 'G')};
  for (int level : {-1, 0, 1, 6, 9}) {
    SCOPED_TRACE(level);
    const std::string path = ::testing::TempDir() + "example-writer-level-" +
                             std::to_string(level) + ".tfrecord.gz";
    ExampleWriter writer(path, ExampleFormat::kAuto, level);
    ASSERT_TRUE(writer.status().ok());
    for (const auto& record : records) {
      ASSERT_TRUE(writer.Add(record));
    }
    ASSERT_TRUE(writer.Close());
    auto reader = TFRecordReader::New(path, "GZIP");
    ASSERT_NE(reader, nullptr);
    for (const auto& record : records) {
      ASSERT_TRUE(reader->GetNext());
      EXPECT_EQ(reader->record(), record);
    }
    EXPECT_FALSE(reader->GetNext());
  }
}

}  // namespace nucleus
