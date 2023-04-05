use crate::silent_ot::get_reg_noise_weight;
use crate::silent_ot::pprf::PprfConfig;
use libote_sys::SilverCode;
use num_integer::Integer;

#[derive(Debug)]
pub struct SilverEncoder {
    pub(crate) enc: libote_sys::SilverEncoder,
    pub(crate) conf: SilverConf,
}

#[derive(Debug, Copy, Clone)]
pub struct SilverConf {
    #[allow(unused)]
    pub(crate) code: libote_sys::SilverCode,
    pub(crate) gap: usize,
    /// the number of OTs being requested.
    pub(crate) requested_num_ots: usize,
    /// The dense vector size, this will be at least as big as mRequestedNumOts.
    pub(crate) N: usize,
    /// The sparse vector size, this will be mN * mScaler.
    pub(crate) N2: usize,
    /// The size of each regular section of the sparse vector.
    pub(crate) size_per: usize,
    /// The number of regular section of the sparse vector.
    pub(crate) num_partitions: usize,
}

impl SilverConf {
    pub fn configure(num_ots: usize, weight: usize, sec_param: usize) -> Self {
        let code = match weight {
            5 => SilverCode::Weight5,
            11 => SilverCode::Weight11,
            _ => panic!("Unsupported weight."),
        };
        let scaler = 2;
        let gap = code.gap() as usize;
        let num_partitions = get_reg_noise_weight(0.2, sec_param) as usize;
        let size_per = Integer::next_multiple_of(
            &((num_ots * scaler + num_partitions - 1) / num_partitions),
            &8,
        );
        let N2 = size_per * num_partitions + gap;
        let N = N2 / scaler;
        Self {
            code,
            gap,
            requested_num_ots: num_ots,
            N,
            N2,
            size_per,
            num_partitions,
        }
    }
}

impl From<SilverConf> for PprfConfig {
    fn from(value: SilverConf) -> Self {
        PprfConfig::new(value.size_per, value.num_partitions)
    }
}
