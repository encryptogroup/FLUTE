#pragma once
#include <libOTe/Tools/LDPC/LdpcEncoder.h>
#include <cryptoTools/Common/Defines.h>
#include <rust/cxx.h>

namespace osuCryptoBridge {
    using SilverCodeWeight = osuCrypto::SilverCode::code;

    struct SilverEncBridge : public oc::SilverEncoder {

        void dualEncodeBlock(rust::Slice<oc::block> c) {
            dualEncode(oc::span<oc::block>(c.begin(), c.end()));
        }

        void dualEncode2Block(rust::Slice<oc::block> c0, rust::Slice<oc::u8> c1) {
            oc::span<oc::block> c0_span(c0.begin(), c0.end());
            oc::span<oc::u8> c1_span(c1.begin(), c1.end());
            dualEncode2(c0_span, c1_span);
        }
    };

    std::unique_ptr<SilverEncBridge> newEnc() {
        return std::make_unique<SilverEncBridge>();
    }
}

