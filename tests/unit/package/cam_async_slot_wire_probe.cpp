// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "ring_layout.h"

using namespace Cam::AsyncPipeline;
constexpr uint64_t DOMAIN = 73;
constexpr uint64_t PAYLOAD_CAPACITY = 193;
constexpr uint64_t METADATA_CAPACITY = 65;
constexpr uint64_t REGISTERED_BYTES = 8192;
struct WordBackend {
    static uint64_t ReadWord(const uint64_t *p, uint64_t i) { return p[i]; }
    static void WriteWord(uint64_t *p, uint64_t i, uint64_t v) { p[i] = v; }
};
void Require(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
RingLayout Layout()
{
    RingLayout layout;
    Require(BuildRingLayout({DOMAIN, DEFAULT_SLOT_COUNT, PAYLOAD_CAPACITY, METADATA_CAPACITY},
        REGISTERED_BYTES, layout) == WireStatus::Success, "layout rejected");
    return layout;
}
SlotControlRecord Ready()
{
    return {PROTOCOL_MAGIC, PROTOCOL_VERSION, PROTOCOL_FINGERPRINT, DOMAIN, 1, 19, 27, 3,
        static_cast<uint64_t>(SlotState::Ready), SOURCE_CONSUMED | REMOTE_VISIBLE,
        PAYLOAD_CAPACITY, METADATA_CAPACITY, DEFAULT_SLOT_COUNT, 0, 0, 0};
}
void Geometry()
{
    auto layout = Layout();
    Require(layout.controlsOffset == RING_ALIGNMENT, "control aliases header");
    Require(layout.payloadOffset == RING_HEADER_BYTES + DEFAULT_SLOT_COUNT * CONTROL_RECORDS_PER_SLOT
        * sizeof(SlotControlRecord), "payload aliases records");
    Require(layout.metadataOffsetInSlot == 256 && layout.slotStride == 384, "padding changed");
    uint64_t previousEnd = RING_HEADER_BYTES;
    for (uint64_t slot = 0; slot < layout.slotCount; ++slot) {
        for (RecordRole role : {RecordRole::ProducerReady, RecordRole::ReceiverCredit}) {
            uint64_t offset;
            Require(ControlOffset(layout, slot, role, offset) == WireStatus::Success, "control offset");
            Require(offset >= previousEnd && offset % RING_ALIGNMENT == 0, "shared control line");
            previousEnd = offset + sizeof(SlotControlRecord);
        }
    }
    Require(previousEnd == layout.payloadOffset, "control end");
    uint64_t first, second;
    Require(PayloadOffset(layout, 0, 0, PAYLOAD_CAPACITY, first) == WireStatus::Success, "first payload");
    Require(PayloadOffset(layout, 1, 0, PAYLOAD_CAPACITY, second) == WireStatus::Success, "second payload");
    Require(second - first == layout.slotStride && first + PAYLOAD_CAPACITY < second, "slots overlap");
}
void Rejections()
{
    RingLayout layout;
    Require(BuildRingLayout({0, 2, 0, 0}, REGISTERED_BYTES, layout) == WireStatus::InvalidLayout, "zero domain");
    Require(BuildRingLayout({DOMAIN, 1, 0, 0}, REGISTERED_BYTES, layout) == WireStatus::InvalidLayout, "one slot");
    Require(BuildRingLayout({DOMAIN, MAX_SLOT_COUNT + 1, 0, 0}, REGISTERED_BYTES, layout)
        == WireStatus::InvalidLayout, "too many slots");
    Require(BuildRingLayout({DOMAIN, 2, std::numeric_limits<uint64_t>::max(), 0},
        std::numeric_limits<uint64_t>::max(), layout) == WireStatus::Overflow, "align overflow");
    auto valid = Layout();
    Require(BuildRingLayout({DOMAIN, 2, PAYLOAD_CAPACITY, METADATA_CAPACITY}, valid.totalBytes - 1, layout)
        == WireStatus::InvalidLayout, "window too short");
    uint64_t offset;
    Require(ControlOffset(valid, valid.slotCount, RecordRole::ProducerReady, offset)
        == WireStatus::InvalidLayout, "bad slot");
    Require(PayloadOffset(valid, 0, PAYLOAD_CAPACITY, 1, offset) == WireStatus::InvalidLayout, "payload overrun");
}
void Codec()
{
    auto ready = Ready();
    std::array<uint64_t, sizeof(SlotControlRecord) / sizeof(uint64_t)> words{};
    EncodeControlWords<WordBackend>(ready, words.data());
    Require(words[0] == PROTOCOL_MAGIC && words[1] == PROTOCOL_VERSION
        && words[2] == PROTOCOL_FINGERPRINT && words[5] == ready.epoch
        && words[8] == ready.state && words[12] == DEFAULT_SLOT_COUNT, "field positions");
    SlotControlRecord decoded{};
    DecodeControlWords<WordBackend>(words.data(), decoded);
    Require(std::memcmp(&ready, &decoded, sizeof(ready)) == 0, "codec changed words");
    const Lease lease{DOMAIN, 1, 19, 27, 3};
    Require(ValidateWireRecord(decoded, Layout(), lease, RecordRole::ProducerReady) == WireStatus::Success, "ready");
    Require(ValidateWireRecord(decoded, Layout(), lease, RecordRole::ReceiverCredit)
        == WireStatus::MissingBoundary, "ready mistaken for credit");
    decoded.state = static_cast<uint64_t>(SlotState::Consumed);
    decoded.boundaryFlags |= RECEIVER_CONSUMED | RECEIVER_CREDIT;
    Require(ValidateWireRecord(decoded, Layout(), lease, RecordRole::ReceiverCredit) == WireStatus::Success, "credit");
    // A syntactically valid credit does not imply TaskRetired or mutate a pool.
    Require((decoded.boundaryFlags & TASK_RETIRED) == 0, "codec invented retirement");
}
void BadRecords()
{
    auto record = Ready();
    const Lease expected{DOMAIN, 1, 19, 27, 3};
    auto Check = [&](const SlotControlRecord &r) {
        return ValidateWireRecord(r, Layout(), expected, RecordRole::ProducerReady);
    };
    auto bad = record; bad.magic = 0; Require(Check(bad) == WireStatus::InvalidSchema, "compact accepted");
    bad = record; ++bad.version; Require(Check(bad) == WireStatus::InvalidSchema, "version");
    bad = record; ++bad.fingerprint; Require(Check(bad) == WireStatus::InvalidSchema, "fingerprint");
    bad = record; ++bad.epoch; Require(Check(bad) == WireStatus::WrongLease, "epoch");
    bad = record; ++bad.requestId; Require(Check(bad) == WireStatus::WrongLease, "request");
    bad = record; bad.payloadBytes = PAYLOAD_CAPACITY + 1; Require(Check(bad) == WireStatus::InvalidSchema, "length");
    bad = record; bad.reserved1 = 1; Require(Check(bad) == WireStatus::InvalidSchema, "reserved");
    bad = record; bad.boundaryFlags = SOURCE_CONSUMED; Require(Check(bad) == WireStatus::InvalidSchema, "visibility");
}
int main(int argc, char **argv)
{
    try {
        Require(argc == 2, "one scenario required");
        const std::string scenario = argv[1];
        if (scenario == "geometry") Geometry();
        else if (scenario == "rejections") Rejections();
        else if (scenario == "codec") Codec();
        else if (scenario == "bad_records") BadRecords();
        else throw std::runtime_error("unknown scenario");
        std::cout << scenario << " passed\n";
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
