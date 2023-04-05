use std::fmt::{Debug, Formatter};
// use autocxx::include_cpp;

// include_cpp!(
//     #include "SilverBridge.h"
//     // generate!("osuCrypto::SilverCode")
//     // generate!("osuCryptoBridge::SilverEncBridge")
//     // generate!("osuCrypto::details::SilverRightEncoder")
//     safety!(unsafe)
// );

use cxx::{ExternType, type_id, UniquePtr};
use crate::ffi::SilverEncBridge;

pub struct SilverEncoder {
    inner: UniquePtr<SilverEncBridge>
}

// TODO: Safety...
unsafe impl Send for SilverEncoder {}
unsafe impl Sync for SilverEncoder {}

#[derive(Copy, Clone, Debug)]
pub enum SilverCode {
    Weight5 = 5,
    Weight11 = 11
}

impl SilverEncoder {
    pub fn new(code: SilverCode, rows: u64) -> Self {
        let mut inner = ffi::new_enc();
        inner.pin_mut().init(rows, code.into());
        Self { inner }
    }

    pub fn dual_encode(&mut self, c: &mut [Block]) {
        self.inner.pin_mut().dual_encode_block(c);
    }

    pub fn dual_encode2(&mut self, c0: &mut [Block], c1: &mut [u8]) {
        self.inner.pin_mut().dual_encode2_block(c0, c1);
    }
}

impl Debug for SilverEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SilverEncoder").finish()
    }
}

impl SilverCode {
    pub fn gap(&self) -> u64 {
        ffi::SilverCode::from(*self).gap()
    }
}

impl From<SilverCode> for ffi::SilverCode {
    fn from(value: SilverCode) -> Self {
        match value {
            SilverCode::Weight5 => {ffi::SilverCode {weight: ffi::SilverCodeWeight::Weight5}}
            SilverCode::Weight11 => {ffi::SilverCode {weight: ffi::SilverCodeWeight::Weight11}}
        }
    }
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug, Default, Eq, PartialEq)]
#[repr(C, align(16))]
pub struct Block {
    data: u128
}

unsafe impl ExternType for Block {
    type Id = type_id!("osuCrypto::block");
    type Kind = cxx::kind::Trivial;
}

#[cxx::bridge(namespace = "osuCryptoBridge")]
mod ffi {

    struct SilverCode {
        weight: SilverCodeWeight
    }

    #[repr(u32)]
    enum SilverCodeWeight {
        Weight5 = 5,
        Weight11 = 11
    }

    unsafe extern "C++" {
        include!("libote-sys/src/SilverBridge.h");

        type SilverEncBridge;

        type SilverCodeWeight;
        #[namespace = "osuCrypto"]
        type SilverCode;


        #[namespace = "osuCrypto"]
        #[cxx_name = "block"]
        type Block = super::Block;

        #[cxx_name = "newEnc"]
        fn new_enc() -> UniquePtr<SilverEncBridge>;

        #[cxx_name = "dualEncodeBlock"]
        fn dual_encode_block(self: Pin<&mut SilverEncBridge>, c: &mut [Block]);

        #[cxx_name = "dualEncode2Block"]
        fn dual_encode2_block(self: Pin<&mut SilverEncBridge>, c0: &mut [Block], c1: &mut [u8]);

        fn init(self: Pin<&mut SilverEncBridge>, rows: u64, code: SilverCode);

        #[namespace = "osuCrypto"]
        fn gap(self: &mut SilverCode) -> u64;
    }
}


#[cfg(test)]
mod tests {
    use crate::{Block, ffi};
    use crate::ffi::{SilverCode, SilverCodeWeight};

    #[test]
    fn create_silver_encoder() {
        let enc = ffi::new_enc();
    }

    #[test]
    fn init_encoder() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(50, SilverCode { weight: SilverCodeWeight::Weight5});
    }

    #[test]
    fn dual_encode() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(50, SilverCode { weight: SilverCodeWeight::Weight5});
        let mut c = vec![Block::default(); 100];
        enc.pin_mut().dual_encode_block(&mut c);
    }

    #[test]
    fn dual_encode2() {
        let mut enc = ffi::new_enc();
        enc.pin_mut().init(50, SilverCode { weight: SilverCodeWeight::Weight5});
        let mut c0 = vec![Block::default(); 100];
        let mut c1 = vec![Block::default(); 100];
        enc.pin_mut().dual_encode2_block(&mut c0, &mut c1);
    }

}